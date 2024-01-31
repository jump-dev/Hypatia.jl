#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

"""
Interior point algorithms for conic models.
"""
module Solvers

using DocStringExtensions
using Printf
using LinearAlgebra
using SparseArrays
import SuiteSparse
import LinearMaps
import IterativeSolvers
using Test
import Base.convert
import Hypatia.Cones
import Hypatia.Models
import Hypatia.SparseNonSymCache
import Hypatia.SparseSymCache
import Hypatia.diag_min
import Hypatia.update_fact
import Hypatia.inv_prod
import Hypatia.free_memory
import Hypatia.int_type
import Hypatia.nonsymm_fact_copy!
import Hypatia.symm_fact_copy!
import Hypatia.posdef_fact_copy!
import Hypatia.outer_prod!

const RealOrNothing = Union{Real, Nothing}

include("point.jl")

# solver termination status codes
@enum Status begin
    NotLoaded
    Loaded
    Modified
    SolveCalled
    PrimalInconsistent
    DualInconsistent
    Optimal
    PrimalInfeasible
    DualInfeasible
    IllPosed
    NearOptimal
    NearPrimalInfeasible
    NearDualInfeasible
    NearIllPosed
    SlowProgress
    IterationLimit
    TimeLimit
    NumericalFailure
    UnknownStatus
end

const infeas_statuses =
    (PrimalInfeasible, DualInfeasible, NearPrimalInfeasible, NearDualInfeasible)

# statuses for which near-convergence should be checked
const check_near_statuses = (SlowProgress, IterationLimit, TimeLimit, NumericalFailure)

convert(::Type{String}, status::Status) = string(status)

abstract type Stepper{T <: Real} end

abstract type SystemSolver{T <: Real} end

"""
$(TYPEDEF)

Hypatia's interior point solver type. See source for options and defaults.
"""
mutable struct Solver{T <: Real}
    # main options
    verbose::Bool
    iter_limit::Int
    time_limit::Float64
    tol_rel_opt::T
    tol_abs_opt::T
    tol_feas::T
    tol_infeas::T
    tol_tau_infeas::T
    tol_illposed::T
    tol_slow::T
    near_factor::T
    preprocess::Bool
    reduce::Bool
    rescale::Bool
    init_use_indirect::Bool
    init_tol_qr::T
    tol_inconsistent::T
    stepper::Stepper{T}
    syssolver::SystemSolver{T}
    use_dense_model::Bool # make the model use dense A and G data instead of sparse

    # current status of the solver object and info
    status::Status
    solve_time::Float64
    num_iters::Int

    # helpful performance metrics for major subprocedures
    time_rescale::Float64 # rescale affine data
    time_initx::Float64 # preprocess dual equalities and finding initial x point
    time_inity::Float64 # preprocess primal equalities and finding initial y point
    time_loadsys::Float64 # initialize/load system solver
    time_upsys::Float64 # update LHS and factorization etc for directions solving
    time_upfact::Float64 # update inner factorization for directions solving only
    time_uprhs::Float64 # update RHSs for directions
    time_getdir::Float64 # solve for directions
    time_search::Float64 # searches for alpha parameters

    # model and preprocessed model data
    orig_model::Models.Model{T}
    model::Models.Model{T}
    x_keep_idxs::AbstractVector{Int}
    y_keep_idxs::AbstractVector{Int}
    Ap_fact::Factorization{T}
    Ap_rank::Int
    Ap_R::UpperTriangular{T, <:AbstractMatrix{T}}
    # Ap_Q::Union{UniformScaling, AbstractMatrix{T}}
    Ap_Q::Any  # Any is needed for Julia 1.10 and later
    AG_fact::Factorization{T}
    AG_rank::Int
    AG_R::UpperTriangular{T, <:AbstractMatrix{T}}
    reduce_cQ1::Any
    reduce_Rpib0::Any
    reduce_GQ1::Any
    reduce_Ap_R::Any
    reduce_Ap_Q::Any
    reduce_y_keep_idxs::Any
    reduce_row_piv_inv::Any

    # current iterate
    point::Point{T}
    mu::T

    # result (solution) point
    result::Point{T}

    # residuals
    x_residual::Vector{T}
    y_residual::Vector{T}
    z_residual::Vector{T}
    tau_residual::T
    x_norm_res_t::T
    y_norm_res_t::T
    z_norm_res_t::T
    x_norm_res::T
    y_norm_res::T
    z_norm_res::T

    # direction solving helpers
    res_norm_cutoff::T # desired worst-case residual norm (inf) on direction
    max_ref_steps::Int # maximum number of iterative refinement steps

    # convergence parameters
    primal_obj_t::T
    dual_obj_t::T
    primal_obj::T
    dual_obj::T
    gap::T
    x_feas::T
    y_feas::T
    z_feas::T
    tau_feas::T

    # termination condition helpers
    x_conv_tol::T
    y_conv_tol::T
    z_conv_tol::T
    # TODO count how many slow with counter field and use option for max slow count
    prev_is_slow::Bool
    prev2_is_slow::Bool
    worst_dir_res::T

    # data scaling
    used_rescaling::Bool
    b_scale::Vector{T}
    c_scale::Vector{T}
    h_scale::Vector{T}

    function Solver{T}(;
        verbose::Bool = true,
        iter_limit::Int = 1000,
        time_limit::Real = Inf,
        tol_rel_opt::RealOrNothing = nothing,
        tol_abs_opt::RealOrNothing = nothing,
        tol_feas::RealOrNothing = nothing,
        tol_infeas::RealOrNothing = nothing,
        tol_tau_infeas::RealOrNothing = nothing,
        tol_illposed::RealOrNothing = nothing,
        default_tol_power::RealOrNothing = nothing,
        default_tol_relax::RealOrNothing = nothing,
        tol_slow::Real = 1e-3,
        near_factor::Real = 1000,
        preprocess::Bool = true,
        reduce::Bool = true,
        rescale::Bool = true,
        init_use_indirect::Bool = false,
        init_tol_qr::Real = 1000 * eps(T),
        tol_inconsistent::Real = init_tol_qr,
        stepper::Stepper{T} = CombinedStepper{T}(),
        syssolver::SystemSolver{T} = QRCholDenseSystemSolver{T}(),
        use_dense_model::Bool = (syssolver isa QRCholDenseSystemSolver{T}),
    ) where {T <: Real}
        if isa(syssolver, QRCholSystemSolver{T})
            @assert preprocess # require preprocessing for QRCholSystemSolver # TODO only need primal eq preprocessing or reduction
        end
        if reduce
            @assert preprocess # cannot use reduction without preprocessing # TODO only need primal eq preprocessing
        end
        @assert !(init_use_indirect && preprocess) # cannot use preprocessing and indirect methods for initial point
        @assert near_factor >= 1 # factor to relax tolerances by if fail to converge should be at least 1

        if isnothing(default_tol_power)
            default_tol_power = (T <: LinearAlgebra.BlasReal ? 0.5 : 0.4)
        end
        default_tol_power = T(default_tol_power)
        default_tol_loose = eps(T)^default_tol_power
        default_tol_tight = eps(T)^(T(1.5) * default_tol_power)
        if !isnothing(default_tol_relax)
            default_tol_loose *= T(default_tol_relax)
            default_tol_tight *= T(default_tol_relax)
        end
        if isnothing(tol_rel_opt)
            tol_rel_opt = default_tol_loose
        end
        if isnothing(tol_abs_opt)
            tol_abs_opt = default_tol_tight
        end
        if isnothing(tol_feas)
            tol_feas = default_tol_loose
        end
        if isnothing(tol_infeas)
            tol_infeas = default_tol_tight
        end
        if isnothing(tol_tau_infeas)
            tol_tau_infeas = T(1e-2)
        end
        if isnothing(tol_illposed)
            tol_illposed = default_tol_tight
        end
        @assert min(
            tol_rel_opt,
            tol_abs_opt,
            tol_feas,
            tol_infeas,
            tol_illposed,
            tol_slow,
        ) >= 0

        solver = new{T}()

        solver.verbose = verbose
        solver.iter_limit = iter_limit
        solver.time_limit = time_limit
        solver.tol_rel_opt = tol_rel_opt
        solver.tol_abs_opt = tol_abs_opt
        solver.tol_feas = tol_feas
        solver.tol_infeas = tol_infeas
        solver.tol_tau_infeas = tol_tau_infeas
        solver.tol_illposed = tol_illposed
        solver.tol_slow = tol_slow
        solver.near_factor = near_factor
        solver.preprocess = preprocess
        solver.reduce = reduce
        solver.rescale = rescale
        solver.init_use_indirect = init_use_indirect
        solver.init_tol_qr = init_tol_qr
        solver.tol_inconsistent = tol_inconsistent
        solver.stepper = stepper
        solver.syssolver = syssolver
        solver.status = NotLoaded
        solver.use_dense_model = use_dense_model

        return solver
    end
end

function load(solver::Solver{T}, model::Models.Model{T}) where {T <: Real}
    solver.orig_model = model
    solver.status = Loaded
    return
end

function modify_obj_offset(solver::Solver{T}, offset_new::T) where {T <: Real}
    solver.orig_model.obj_offset = offset_new
    solver.status = Modified
    return
end

function modify_c(solver::Solver{T}, c_new::AbstractVector{T}) where {T <: Real}
    @assert length(c_new) == solver.orig_model.n
    solver.orig_model.c = c_new
    solver.status = Modified
    return
end

function modify_c(
    solver::Solver{T},
    idxs::AbstractVector{Int},
    c_new::Vector{T},
) where {T <: Real}
    @assert length(c_new) == length(idxs) <= solver.orig_model.n
    solver.orig_model.c[idxs] .= c_new
    solver.status = Modified
    return
end

function modify_b(solver::Solver{T}, b_new::AbstractVector{T}) where {T <: Real}
    @assert length(b_new) == solver.orig_model.p
    solver.orig_model.b = b_new
    solver.status = Modified
    return
end

function modify_b(
    solver::Solver{T},
    idxs::AbstractVector{Int},
    b_new::Vector{T},
) where {T <: Real}
    @assert length(b_new) == length(idxs) <= solver.orig_model.p
    solver.orig_model.b[idxs] .= b_new
    solver.status = Modified
    return
end

function modify_h(solver::Solver{T}, h_new::AbstractVector{T}) where {T <: Real}
    @assert length(h_new) == solver.orig_model.q
    solver.orig_model.h = h_new
    solver.status = Modified
    return
end

function modify_h(
    solver::Solver{T},
    idxs::AbstractVector{Int},
    h_new::Vector{T},
) where {T <: Real}
    @assert length(h_new) == length(idxs) <= solver.orig_model.q
    solver.orig_model.h[idxs] .= h_new
    solver.status = Modified
    return
end

function solve(solver::Solver{T}) where {T <: Real}
    init_status = solver.status
    if !in(init_status, (Loaded, Modified))
        @warn("solve called when solver status is $init_status")
        return
    end

    setup_solver(solver)
    if init_status == Modified
        setup_modified(solver)
    else
        setup_loaded(solver)
    end

    if !in(solver.status, (PrimalInconsistent, DualInconsistent))
        initialize_point(solver)
        setup_stepping(solver)
        solver.verbose && print_header(solver.stepper, solver)

        while true
            step_and_check(solver) && break
        end

        if solver.status in check_near_statuses
            check_converged(solver, true)
        end

        postprocess(solver)
    end

    solver.solve_time = time() - solver.solve_time
    if solver.verbose
        println(
            "\nstatus is $(solver.status) after $(solver.num_iters) " *
            "iterations and $(trunc(solver.solve_time, digits=3)) seconds\n",
        )
    end

    free_memory(solver.syssolver)
    flush(stdout)
    return
end

function setup_loaded(solver::Solver{T}) where {T <: Real}
    orig_model = solver.orig_model
    if solver.use_dense_model
        Models.densify!(orig_model)
    end
    solver.result = Point(orig_model)

    # copy original model to solver.model, which may be modified
    model =
        solver.model = Models.Model{T}(
            orig_model.c,
            orig_model.A,
            orig_model.b,
            orig_model.G,
            orig_model.h,
            orig_model.cones,
            obj_offset = orig_model.obj_offset,
        )

    solver.time_rescale = @elapsed solver.used_rescaling = rescale_data(solver)

    setup_point(solver)
    if solver.status != SolveCalled
        return
    end

    load(solver.stepper, solver)
    solver.time_loadsys = @elapsed load(solver.syssolver, solver)

    solver.x_residual = zero(model.c)
    solver.y_residual = zero(model.b)
    solver.z_residual = zero(model.h)
    solver.tau_residual = 0
    return
end

function setup_modified(solver::Solver{T}) where {T <: Real}
    orig_model = solver.orig_model
    model = solver.model

    model.obj_offset = orig_model.obj_offset
    model.c = copy(orig_model.c)
    model.b = copy(orig_model.b)
    model.h = copy(orig_model.h)
    if solver.used_rescaling
        # rescale using scalings computed during first load
        model.c ./= solver.c_scale
        model.b ./= solver.b_scale
        model.h ./= solver.h_scale
    end

    update_initial_point(solver)

    # TODO easier to update these system solvers (just need an update to setup_point_sub):
    @assert solver.syssolver isa Union{QRCholSystemSolver{T}, SymIndefSystemSolver{T}}
    set_point_sub_rhs(solver.syssolver, solver.model)
    return
end

# preprocess and find initial point
function setup_point(solver::Solver{T}) where {T <: Real}
    (init_z, init_s) = initialize_cone_point(solver.orig_model)

    if solver.reduce
        solver.time_inity = @elapsed handle_primal_eq(solver)
        solver.status == PrimalInconsistent && return
        init_y = update_primal_eq(solver, init_z)

        solver.time_initx = @elapsed handle_dual_eq(solver)
        solver.status == DualInconsistent && return
        init_x = update_dual_eq(solver, init_s)
    else
        solver.time_initx = @elapsed handle_dual_eq(solver)
        solver.status == DualInconsistent && return
        init_x = update_dual_eq(solver, init_s)

        solver.time_inity = @elapsed handle_primal_eq(solver)
        solver.status == PrimalInconsistent && return
        init_y = update_primal_eq(solver, init_z)
    end

    point = solver.point = Point(solver.model)
    point.x .= init_x
    point.y .= init_y
    point.z .= init_z
    point.s .= init_s
    return
end

function update_initial_point(solver::Solver{T}) where {T <: Real}
    (init_z, init_s) = initialize_cone_point(solver.orig_model)
    point = solver.point
    if solver.reduce
        point.y .= update_primal_eq(solver, init_z)
        point.x .= update_dual_eq(solver, init_s)
    else
        point.x .= update_dual_eq(solver, init_s)
        point.y .= update_primal_eq(solver, init_z)
    end
    point.z .= init_z
    point.s .= init_s
    return
end

function initialize_point(solver::Solver{T}) where {T <: Real}
    point = solver.point
    point.tau[] = one(T)
    point.kap[] = one(T)

    calc_mu(solver)
    if isnan(solver.mu) || abs(one(T) - solver.mu) > sqrt(eps(T))
        @warn(
            "initial mu is $(solver.mu) but should be 1 (this could " *
            "indicate a problem with cone barrier oracles)"
        )
    end

    model = solver.model
    Cones.load_point.(model.cones, point.primal_views)
    Cones.load_dual_point.(model.cones, point.dual_views)
    return
end

function setup_stepping(solver::Solver{T}) where {T <: Real}
    model = solver.model
    solver.x_conv_tol = inv(1 + norm(model.c, Inf))
    solver.y_conv_tol = inv(1 + norm(model.b, Inf))
    solver.z_conv_tol = inv(1 + norm(model.h, Inf))
    solver.prev_is_slow = false
    solver.prev2_is_slow = false
    solver.worst_dir_res = 0
    return
end

function setup_solver(solver::Solver{T}) where {T <: Real}
    solver.status = SolveCalled
    solver.num_iters = 0
    solver.solve_time = time()

    solver.time_rescale = 0
    solver.time_initx = 0
    solver.time_inity = 0
    solver.time_loadsys = 0
    solver.time_upsys = 0
    solver.time_upfact = 0
    solver.time_uprhs = 0
    solver.time_getdir = 0
    solver.time_search = 0

    solver.res_norm_cutoff = 0
    solver.max_ref_steps = 5

    solver.x_norm_res_t = NaN
    solver.y_norm_res_t = NaN
    solver.z_norm_res_t = NaN
    solver.x_norm_res = NaN
    solver.y_norm_res = NaN
    solver.z_norm_res = NaN

    solver.primal_obj_t = NaN
    solver.dual_obj_t = NaN
    solver.primal_obj = NaN
    solver.dual_obj = NaN
    solver.gap = NaN
    solver.x_feas = NaN
    solver.y_feas = NaN
    solver.z_feas = NaN
    solver.tau_feas = NaN
    return
end

function step_and_check(solver::Solver{T}) where {T <: Real}
    stepper = solver.stepper
    improv = calc_convergence_params(solver)

    if solver.verbose
        print_iteration(stepper, solver)
    end

    check_converged(solver) && return true

    if solver.num_iters == solver.iter_limit
        solver.verbose && println("iteration limit reached; terminating")
        solver.status = IterationLimit
        return true
    end

    if time() - solver.solve_time >= solver.time_limit
        solver.verbose && println("time limit reached; terminating")
        solver.status = TimeLimit
        return true
    end

    if expect_improvement(stepper)
        if improv < solver.tol_slow
            if solver.prev_is_slow && solver.prev2_is_slow
                if solver.verbose
                    println("slow progress in consecutive " * "iterations; terminating")
                end
                solver.status = SlowProgress
                return true
            else
                solver.prev2_is_slow = solver.prev_is_slow
                solver.prev_is_slow = true
            end
        else
            solver.prev2_is_slow = solver.prev_is_slow
            solver.prev_is_slow = false
        end
    end

    solver.res_norm_cutoff =
        T(1e-4) *
        max(solver.x_norm_res, solver.y_norm_res, solver.z_norm_res, solver.tau_feas)
    solver.worst_dir_res = 0

    step(stepper, solver) || return true
    calc_mu(solver)

    if min(solver.point.tau[], solver.point.kap[], solver.mu) <= 0
        @warn("tau, kappa, or mu is nonpositive; terminating")
        solver.status = NumericalFailure
        return true
    end

    flush(stdout)
    solver.num_iters += 1
    return false
end

function calc_mu(solver::Solver{T}) where {T <: Real}
    point = solver.point
    solver.mu = (dot(point.z, point.s) + point.tau[] * point.kap[]) / (solver.model.nu + 1)
    return solver.mu
end

function calc_convergence_params(solver::Solver{T}) where {T <: Real}
    model = solver.model
    point = solver.point
    tau = point.tau[]

    # x_residual = -A'*y - G'*z - c*tau
    x_residual = solver.x_residual
    mul!(x_residual, model.G', point.z)
    mul!(x_residual, model.A', point.y, true, true)
    solver.x_norm_res_t = norm(x_residual, Inf)
    @. x_residual += model.c * tau
    solver.x_norm_res = norm(x_residual, Inf) / tau
    @. x_residual *= -1
    x_feas = solver.x_norm_res * solver.x_conv_tol

    # y_residual = A*x - b*tau
    y_residual = solver.y_residual
    mul!(y_residual, model.A, point.x)
    solver.y_norm_res_t = norm(y_residual, Inf)
    @. y_residual -= model.b * tau
    solver.y_norm_res = norm(y_residual, Inf) / tau
    y_feas = solver.y_norm_res * solver.y_conv_tol

    # z_residual = s + G*x - h*tau
    z_residual = solver.z_residual
    mul!(z_residual, model.G, point.x)
    @. z_residual += point.s
    solver.z_norm_res_t = norm(z_residual, Inf)
    @. z_residual -= model.h * tau
    solver.z_norm_res = norm(z_residual, Inf) / tau
    z_feas = solver.z_norm_res * solver.z_conv_tol

    # tau_residual = c'*x + b'*y + h'*z + kap
    solver.primal_obj_t = dot(model.c, point.x)
    solver.dual_obj_t = -dot(model.b, point.y) - dot(model.h, point.z)
    solver.tau_residual = solver.primal_obj_t - solver.dual_obj_t + point.kap[]
    tau_feas = abs(solver.tau_residual)

    # check improvement
    improv = zero(T)
    for (curr, prev) in (
        (x_feas, solver.x_feas),
        (y_feas, solver.y_feas),
        (z_feas, solver.z_feas),
        (tau_feas, solver.tau_feas),
    )
        if isnan(prev) || isnan(curr)
            continue
        end
        improv = max(improv, (prev - curr) / (abs(prev) + eps(T)))
    end
    solver.x_feas = x_feas
    solver.y_feas = y_feas
    solver.z_feas = z_feas
    solver.tau_feas = tau_feas

    # gap
    solver.primal_obj = solver.primal_obj_t / tau + model.obj_offset
    solver.dual_obj = solver.dual_obj_t / tau + model.obj_offset
    solver.gap = dot(point.z, point.s)

    return improv
end

# check convergence criteria, with relaxed tolerances if check_near
function check_converged(solver::Solver{T}, check_near::Bool = false) where {T <: Real}
    near_factor = (check_near ? solver.near_factor : one(T))
    tau = solver.point.tau[]
    primal_obj_t = solver.primal_obj_t
    dual_obj_t = solver.dual_obj_t

    # check optimality
    worst_feas = max(solver.x_feas, solver.y_feas, solver.z_feas)
    is_feas = (worst_feas <= near_factor * solver.tol_feas)
    if is_feas
        is_abs_opt = (solver.gap <= near_factor * solver.tol_abs_opt)
        worst_gap = min(solver.gap / tau, abs(primal_obj_t - dual_obj_t))
        max_tau_obj = max(tau, min(abs(primal_obj_t), abs(dual_obj_t)))
        is_rel_opt = (worst_gap <= near_factor * solver.tol_rel_opt * max_tau_obj)
        if is_abs_opt || is_rel_opt
            solver.verbose && println("optimal solution found; terminating")
            solver.status = (check_near ? NearOptimal : Optimal)
            return true
        end
    end

    # check primal or dual infeasibility
    if tau <= near_factor * solver.tol_tau_infeas
        tol = near_factor * solver.tol_infeas

        if (dual_obj_t > eps(T)) && (solver.x_norm_res_t <= tol * dual_obj_t)
            solver.verbose && println("primal infeasibility detected; terminating")
            solver.status = (check_near ? NearPrimalInfeasible : PrimalInfeasible)
            solver.primal_obj = primal_obj_t
            solver.dual_obj = dual_obj_t
            return true
        end

        yz_res = max(solver.y_norm_res_t, solver.z_norm_res_t)
        if (primal_obj_t < -eps(T)) && (yz_res <= tol * -primal_obj_t)
            solver.verbose && println("dual infeasibility detected; terminating")
            solver.status = (check_near ? NearDualInfeasible : DualInfeasible)
            solver.primal_obj = primal_obj_t
            solver.dual_obj = dual_obj_t
            return true
        end
    end

    # check ill-posedness
    if max(tau, solver.point.kap[]) <= near_factor * solver.tol_illposed
        solver.verbose && println("ill-posedness detected; terminating")
        solver.status = (check_near ? NearIllPosed : IllPosed)
        return true
    end

    return false
end

function initialize_cone_point(model::Models.Model{T}) where {T <: Real}
    init_z = zeros(T, model.q)
    init_s = zeros(T, model.q)

    for (cone, idxs) in zip(model.cones, model.cone_idxs)
        Cones.setup_data!(cone)
        primal_k = view(Cones.use_dual_barrier(cone) ? init_z : init_s, idxs)
        dual_k = view(Cones.use_dual_barrier(cone) ? init_s : init_z, idxs)
        Cones.set_initial_point!(primal_k, cone)
        Cones.load_point(cone, primal_k)
        @assert Cones.is_feas(cone)
        g = Cones.grad(cone)
        @. dual_k = -g
        Cones.load_dual_point(cone, dual_k)
        @assert Cones.is_dual_feas(cone)
    end

    return (init_z, init_s)
end

get_model(solver::Solver) = solver.orig_model

get_status(solver::Solver) = solver.status
get_solve_time(solver::Solver) = solver.solve_time
get_num_iters(solver::Solver) = solver.num_iters

get_primal_obj(solver::Solver) = solver.primal_obj
get_dual_obj(solver::Solver) = solver.dual_obj

get_s(solver::Solver) = copy(solver.result.s)
get_z(solver::Solver) = copy(solver.result.z)
get_x(solver::Solver) = copy(solver.result.x)
get_y(solver::Solver) = copy(solver.result.y)

get_tau(solver::Solver) = solver.point.tau[]
get_kappa(solver::Solver) = solver.point.kap[]
get_mu(solver::Solver) = solver.mu

include("process.jl")

include("search.jl")

include("steppers/common.jl")

include("systemsolvers/common.jl")

# release memory used by sparse system solvers
free_memory(::SystemSolver) = nothing
function free_memory(syssolver::Union{NaiveSparseSystemSolver, SymIndefSparseSystemSolver})
    return free_memory(syssolver.fact_cache)
end

# verbose helpers
function print_header(stepper::Stepper, solver::Solver)
    println()
    @printf("%5s %12s %12s |%9s ", "iter", "p_obj", "d_obj", "abs_gap")
    if iszero(solver.model.p)
        @printf("%9s %9s ", "x_feas", "z_feas")
    else
        @printf("%9s %9s %9s ", "x_feas", "y_feas", "z_feas")
    end
    @printf("|%9s %9s %9s |%8s %8s ", "tau", "kap", "mu", "dir_res", "prox")

    print_header_more(stepper, solver)
    println()
    flush(stdout)
    return
end
print_header_more(stepper::Stepper, solver::Solver) = nothing

function print_iteration(stepper::Stepper, solver::Solver)
    @printf(
        "%5d %12.4e %12.4e |%9.2e ",
        solver.num_iters,
        solver.primal_obj,
        solver.dual_obj,
        solver.gap
    )
    if iszero(solver.model.p)
        @printf("%9.2e %9.2e ", solver.x_feas, solver.z_feas)
    else
        @printf("%9.2e %9.2e %9.2e ", solver.x_feas, solver.y_feas, solver.z_feas)
    end
    @printf("|%9.2e %9.2e %9.2e |", solver.point.tau[], solver.point.kap[], solver.mu)

    if !iszero(solver.num_iters)
        @printf("%8.1e %8.1e ", solver.worst_dir_res, stepper.searcher.prox)
        print_iteration_more(stepper, solver)
    end
    println()
    flush(stdout)
    return
end
print_iteration_more(stepper::Stepper, solver::Solver) = nothing

end
