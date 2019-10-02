#=
Copyright 2018, Chris Coey and contributors

functions and caches for interior point algorithms
=#

module Solvers

using Printf
using LinearAlgebra
using SparseArrays
import SuiteSparse
import IterativeSolvers
using Test
using TimerOutputs
import Hypatia.Cones
import Hypatia.Models
# import Hypatia.HypLUSolveCache
# import Hypatia.hyp_lu_solve!
# import Hypatia.HypBKSolveCache
# import Hypatia.hyp_bk_solve!
# import Hypatia.HypCholSolveCache
# import Hypatia.hyp_chol_solve!
# import Hypatia.set_min_diag!
import Hypatia.BlockMatrix
import Hypatia.SparseNonSymCache
import Hypatia.SparseSymCache
import Hypatia.update_sparse_fact
import Hypatia.solve_sparse_system
import Hypatia.free_memory
import Hypatia.int_type
import Hypatia.GESVXNonSymCache
import Hypatia.solve_dense_system
import Hypatia.reset_fact

abstract type Stepper{T <: Real} end

abstract type SystemSolver{T <: Real} end

mutable struct Solver{T <: Real}
    # main options
    verbose::Bool
    iter_limit::Int
    time_limit::Float64
    tol_rel_opt::T
    tol_abs_opt::T
    tol_feas::T
    tol_slow::T
    preprocess::Bool
    init_use_indirect::Bool
    init_tol_qr::T
    init_use_fallback::Bool
    max_nbhd::T
    use_infty_nbhd::Bool
    stepper::Stepper{T}
    system_solver::SystemSolver{T}

    # current status of the solver object
    status::Symbol

    # solve info and timers
    solve_time::Float64
    timer::TimerOutput
    num_iters::Int

    # model and preprocessed model data
    orig_model::Models.Model{T}
    model::Models.Model{T}
    x_keep_idxs::AbstractVector{Int}
    y_keep_idxs::AbstractVector{Int}
    Ap_R::UpperTriangular{T, <:AbstractMatrix{T}}
    Ap_Q::Union{UniformScaling, AbstractMatrix{T}}

    # current iterate
    point::Models.Point{T}
    tau::T
    kap::T
    mu::T

    # residuals
    x_residual::Vector{T}
    y_residual::Vector{T}
    z_residual::Vector{T}
    x_norm_res_t::T
    y_norm_res_t::T
    z_norm_res_t::T
    x_norm_res::T
    y_norm_res::T
    z_norm_res::T

    # convergence parameters
    primal_obj_t::T
    dual_obj_t::T
    primal_obj::T
    dual_obj::T
    gap::T
    rel_gap::T
    x_feas::T
    y_feas::T
    z_feas::T

    # termination condition helpers
    x_conv_tol::T
    y_conv_tol::T
    z_conv_tol::T
    prev_is_slow::Bool
    prev2_is_slow::Bool
    prev_gap::T
    prev_rel_gap::T
    prev_x_feas::T
    prev_y_feas::T
    prev_z_feas::T

    # step helpers
    keep_iterating::Bool
    prev_aff_alpha::T
    prev_gamma::T
    prev_alpha::T
    z_temp::Vector{T}
    s_temp::Vector{T}
    primal_views
    dual_views
    nbhd_temp
    cones_infeas::Vector{Bool}
    cones_loaded::Vector{Bool}

    function Solver{T}(;
        verbose::Bool = false,
        iter_limit::Int = 1000,
        time_limit::Real = Inf,
        tol_rel_opt::Real = sqrt(eps(T)),
        tol_abs_opt::Real = sqrt(eps(T)),
        tol_feas::Real = sqrt(eps(T)),
        tol_slow::Real = 1e-3,
        preprocess::Bool = true,
        init_use_indirect::Bool = false,
        init_tol_qr::Real = 100 * eps(T),
        init_use_fallback::Bool = true,
        max_nbhd::Real = 0.7,
        use_infty_nbhd::Bool = true,
        stepper::Stepper{T} = CombinedStepper{T}(),
        system_solver::SystemSolver{T} = QRCholDenseSystemSolver{T}(),
        ) where {T <: Real}
        if isa(system_solver, QRCholSystemSolver{T})
            @assert preprocess # require preprocessing for QRCholSystemSolver
        end

        solver = new{T}()
        solver.verbose = verbose
        solver.iter_limit = iter_limit
        solver.time_limit = time_limit
        solver.tol_rel_opt = tol_rel_opt
        solver.tol_abs_opt = tol_abs_opt
        solver.tol_feas = tol_feas
        solver.tol_slow = tol_slow
        solver.preprocess = preprocess
        solver.init_use_indirect = init_use_indirect
        solver.init_tol_qr = init_tol_qr
        solver.init_use_fallback = init_use_fallback
        solver.max_nbhd = max_nbhd
        solver.use_infty_nbhd = use_infty_nbhd
        solver.stepper = stepper
        solver.system_solver = system_solver
        solver.status = :NotLoaded

        return solver
    end
end

get_timer(solver::Solver) = solver.timer

get_status(solver::Solver) = solver.status
get_solve_time(solver::Solver) = solver.solve_time
get_num_iters(solver::Solver) = solver.num_iters

get_primal_obj(solver::Solver) = solver.primal_obj
get_dual_obj(solver::Solver) = solver.dual_obj

get_s(solver::Solver) = copy(solver.point.s)
get_z(solver::Solver) = copy(solver.point.z)

function get_x(solver::Solver{T}) where {T <: Real}
    if solver.preprocess
        x = zeros(T, solver.orig_model.n)
        x[solver.x_keep_idxs] = solver.point.x # unpreprocess solver's solution
    else
        x = copy(solver.point.x)
    end
    return x
end

function get_y(solver::Solver{T}) where {T <: Real}
    if solver.preprocess
        y = zeros(T, solver.orig_model.p)
        y[solver.y_keep_idxs] = solver.point.y # unpreprocess solver's solution
    else
        y = copy(solver.point.y)
    end
    return y
end

get_tau(solver::Solver) = solver.tau
get_kappa(solver::Solver) = solver.kap
get_mu(solver::Solver) = solver.mu

function load(solver::Solver{T}, model::Models.Model{T}) where {T <: Real}
    # @assert solver.status == :NotLoaded # TODO maybe want a reset function that just keeps options
    solver.orig_model = model
    solver.status = :Loaded
    return solver
end

# solve, optionally test conic certificates, and return solve information
function solve_check(
    model::Models.Model{T};
    solver::Solver{T} = Solver{T}(),
    test::Bool = true,
    atol::Real = sqrt(sqrt(eps(T))),
    rtol::Real = sqrt(sqrt(eps(T))),
    ) where {T <: Real}
    load(solver, model)
    solve(solver)

    status = get_status(solver)
    primal_obj = get_primal_obj(solver)
    dual_obj = get_dual_obj(solver)
    x = get_x(solver)
    y = get_y(solver)
    s = get_s(solver)
    z = get_z(solver)

    if test
        (c, A, b, G, h, cones, obj_offset) = (model.c, model.A, model.b, model.G, model.h, model.cones, model.obj_offset)
        if status == :Optimal
            @test primal_obj ≈ dual_obj atol=atol rtol=rtol
            @test A * x ≈ b atol=atol rtol=rtol
            @test G * x + s ≈ h atol=atol rtol=rtol
            @test G' * z + A' * y ≈ -c atol=atol rtol=rtol
            @test dot(c, x) + obj_offset ≈ primal_obj atol=atol^2 rtol=rtol^2
            @test -dot(b, y) - dot(h, z) + obj_offset ≈ dual_obj atol=atol^2 rtol=rtol^2
            @test dot(s, z) ≈ zero(T) atol=10atol rtol=10rtol
        elseif status == :PrimalInfeasible
            @test dual_obj > obj_offset
            @test -dot(b, y) - dot(h, z) + obj_offset ≈ dual_obj atol=atol^2 rtol=rtol^2
            # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
            # @test G' * z ≈ -A' * y atol=atol rtol=rtol
        elseif status == :DualInfeasible
            @test primal_obj < obj_offset
            @test dot(c, x) + obj_offset ≈ primal_obj atol=atol^2 rtol=rtol^2
            # TODO conv check causes us to stop before this is satisfied to sufficient tolerance - maybe add option to keep going
            # @test G * x ≈ -s atol=atol rtol=rtol
            # @test A * x ≈ zeros(T, length(y)) atol=atol rtol=rtol
        elseif status == :IllPosed
            # TODO primal vs dual ill-posed statuses and conditions
        end
    end

    solve_time = get_solve_time(solver)

    return (solver = solver, model = model, solve_time = solve_time, primal_obj = primal_obj, dual_obj = dual_obj, status = status, x = x, y = y, s = s, z = z)
end

# build model and call solve_check
function build_solve_check(
    c::Vector{T},
    A,
    b::Vector{T},
    G,
    h::Vector{T},
    cones::Vector{Cones.Cone{T}};
    obj_offset::T = zero(T),
    other_options...
    ) where {T <: Real}
    model = Models.Model{T}(c, A, b, G, h, cones, obj_offset = obj_offset)
    return solve_check(model; other_options...)
end

include("homogeneous_self_dual/solver.jl")
include("homogeneous_self_dual/stepper.jl")
include("homogeneous_self_dual/naive.jl")
include("homogeneous_self_dual/naiveelim.jl")
include("homogeneous_self_dual/symindef.jl")
include("homogeneous_self_dual/qrchol.jl")

# release memory used by sparse system solvers
free_memory(::SystemSolver) = nothing
free_memory(system_solver::Union{NaiveSparseSystemSolver, SymIndefSparseSystemSolver}) = free_memory(system_solver.fact_cache)

end
