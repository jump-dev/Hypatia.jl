#=
functions and caches for interior point algorithms
=#

module Solvers

using Printf
using LinearAlgebra
using SparseArrays
import SuiteSparse
import LinearMaps
import IterativeSolvers
using Test
import Hypatia.Cones
import Hypatia.Models
import Hypatia.SparseNonSymCache
import Hypatia.SparseSymCache
import Hypatia.update_fact
import Hypatia.inv_prod
import Hypatia.free_memory
import Hypatia.int_type
import Hypatia.DenseNonSymCache
import Hypatia.DenseSymCache
import Hypatia.DensePosDefCache
import Hypatia.load_matrix
import Hypatia.invert
import Hypatia.increase_diag!
import Hypatia.outer_prod

RealOrNothing = Union{Real, Nothing}

include("point.jl")

# solver termination status codes
@enum Status begin
    NotLoaded
    Loaded
    SolveCalled
    Optimal
    PrimalInfeasible
    DualInfeasible
    IllPosed
    PrimalInconsistent
    DualInconsistent
    SlowProgress
    IterationLimit
    TimeLimit
    NumericalFailure
    UnknownStatus
end

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
    tol_infeas::T
    tol_slow::T
    preprocess::Bool
    reduce::Bool
    rescale::Bool
    init_use_indirect::Bool
    init_tol_qr::T
    stepper::Stepper{T}
    system_solver::SystemSolver{T}

    # current status of the solver object and info
    status::Status
    solve_time::Float64
    num_iters::Int

    # model and preprocessed model data
    orig_model::Models.Model{T}
    model::Models.Model{T}
    x_keep_idxs::AbstractVector{Int}
    y_keep_idxs::AbstractVector{Int}
    Ap_R::UpperTriangular{T, <:AbstractMatrix{T}}
    Ap_Q::Union{UniformScaling, AbstractMatrix{T}}
    reduce_cQ1
    reduce_Rpib0
    reduce_GQ1
    reduce_Ap_R
    reduce_Ap_Q
    reduce_y_keep_idxs
    reduce_row_piv_inv
    used_rescaling::Bool
    b_scale::Vector{T}
    c_scale::Vector{T}
    h_scale::Vector{T}

    # points
    point::Point{T}
    mu::T
    best_point::Point{T}

    # convergence goals
    goal_opt::T
    goal_prinf::T
    goal_duinf::T
    goal_ip::T
    goal::T

    # result (solution) point
    result::Point{T}

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
    tau_residual::T

    # convergence parameters
    primal_obj_t::T
    dual_obj_t::T
    primal_obj::T
    dual_obj::T
    gap::T
    x_feas::T
    y_feas::T
    z_feas::T
    worst_feas::T
    slow_count::Int

    # termination condition helpers
    x_conv_tol::T
    y_conv_tol::T
    z_conv_tol::T

    function Solver{T}(;
        verbose::Bool = true,
        iter_limit::Int = 1000,
        time_limit::Real = Inf,
        tol_rel_opt::RealOrNothing = nothing,
        tol_abs_opt::RealOrNothing = nothing,
        tol_feas::RealOrNothing = nothing,
        tol_infeas::RealOrNothing = nothing,
        default_tol_power::RealOrNothing = nothing,
        default_tol_relax::RealOrNothing = nothing,
        tol_slow::Real = 0.01,
        preprocess::Bool = true,
        reduce::Bool = true,
        rescale::Bool = true,
        init_use_indirect::Bool = false,
        init_tol_qr::Real = 1000 * eps(T),
        stepper::Stepper{T} = default_stepper(T),
        system_solver::SystemSolver{T} = default_system_solver(T),
        ) where {T <: Real}
        if isa(system_solver, QRCholSystemSolver{T})
            @assert preprocess # require preprocessing for QRCholSystemSolver # TODO only need primal eq preprocessing or reduction
        end
        if reduce
            @assert preprocess # cannot use reduction without preprocessing # TODO only need primal eq preprocessing
        end
        # @assert !(init_use_indirect && preprocess) # cannot use preprocessing and indirect methods for initial point

        if isnothing(default_tol_power)
            default_tol_power = (T <: LinearAlgebra.BlasReal ? 0.5 : 0.4)
        end
        default_tol_power = T(default_tol_power)
        default_tol_loose = eps(T) ^ default_tol_power
        default_tol_tight = eps(T) ^ (T(1.5) * default_tol_power)
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
        @assert min(tol_rel_opt, tol_abs_opt, tol_feas, tol_infeas, tol_slow) > 0

        solver = new{T}()

        solver.verbose = verbose
        solver.iter_limit = iter_limit
        solver.time_limit = time_limit
        solver.tol_rel_opt = tol_rel_opt
        solver.tol_abs_opt = tol_abs_opt
        solver.tol_feas = tol_feas
        solver.tol_infeas = tol_infeas
        solver.tol_slow = tol_slow
        solver.preprocess = preprocess
        solver.reduce = reduce
        solver.rescale = rescale
        solver.init_use_indirect = init_use_indirect
        solver.init_tol_qr = init_tol_qr
        solver.stepper = stepper
        solver.system_solver = system_solver
        solver.status = NotLoaded

        return solver
    end
end

default_stepper(T) = CombinedStepper{T}()
default_system_solver(T) = QRCholDenseSystemSolver{T}()

function solve(solver::Solver{T}) where {T <: Real}
    @assert solver.status == Loaded
    solver.status = SolveCalled
    start_time = time()

    verbose = solver.verbose
    solver.num_iters = 0
    solver.solve_time = NaN

    solver.x_norm_res_t = NaN
    solver.y_norm_res_t = NaN
    solver.z_norm_res_t = NaN
    solver.x_norm_res = NaN
    solver.y_norm_res = NaN
    solver.z_norm_res = NaN
    solver.tau_residual = NaN

    solver.primal_obj_t = NaN
    solver.dual_obj_t = NaN
    solver.primal_obj = NaN
    solver.dual_obj = NaN
    solver.gap = NaN
    solver.x_feas = NaN
    solver.y_feas = NaN
    solver.z_feas = NaN
    solver.worst_feas = NaN
    solver.slow_count = 0

    solver.goal_opt = NaN
    solver.goal_prinf = NaN
    solver.goal_duinf = NaN
    solver.goal_ip = NaN
    solver.goal = NaN

    orig_model = solver.orig_model
    solver.result = Point(orig_model)

    # preprocess and find initial point
    model = solver.model = Models.Model{T}(orig_model.c, orig_model.A, orig_model.b, orig_model.G, orig_model.h, orig_model.cones, obj_offset = orig_model.obj_offset) # copy original model to solver.model, which may be modified
    (init_z, init_s) = initialize_cone_point(solver.orig_model)
    solver.used_rescaling = rescale_data(solver)
    if solver.reduce
        # TODO don't find point / unnecessary stuff before reduce
        init_y = find_initial_y(solver, init_z, true)
        init_x = find_initial_x(solver, init_s)
    else
        init_x = find_initial_x(solver, init_s)
        init_y = find_initial_y(solver, init_z, false)
    end

    if solver.status == SolveCalled
        point = solver.point = Point(model)
        solver.best_point = Point(model)
        point.x .= init_x
        point.y .= init_y
        point.z .= init_z
        point.s .= init_s
        point.tau[] = one(T)
        point.kap[] = one(T)
        calc_mu(solver)
        if isnan(solver.mu) || abs(one(T) - solver.mu) > sqrt(eps(T))
            @warn("initial mu is $(solver.mu) but should be 1 (this could indicate a problem with cone barrier oracles)")
        end
        Cones.load_point.(model.cones, point.primal_views)
        Cones.load_dual_point.(model.cones, point.dual_views)

        # setup iteration helpers
        solver.x_residual = zero(model.c)
        solver.y_residual = zero(model.b)
        solver.z_residual = zero(model.h)

        solver.x_conv_tol = inv(1 + norm(model.c, Inf))
        solver.y_conv_tol = inv(1 + norm(model.b, Inf))
        solver.z_conv_tol = inv(1 + norm(model.h, Inf))

        stepper = solver.stepper
        load(stepper, solver)
        load(solver.system_solver, solver)

        verbose && print_header(stepper, solver)
        flush(stdout)

        # iterate from initial point
        while true
            calc_residuals(solver)
            impr = update_goals(solver)

            if verbose
                print_iteration(stepper, solver)
                flush(stdout)
            end

            if solver.goal < 1
                # a convergence goal is met, so update best point and finish
                copyto!(solver.best_point.vec, point.vec)
                if solver.goal_opt < 1
                    verbose && println("optimal solution found; terminating")
                    solver.status = Optimal
                elseif solver.goal_prinf < 1
                    verbose && println("primal infeasibility detected; terminating")
                    solver.status = PrimalInfeasible
                    solver.primal_obj = solver.primal_obj_t
                    solver.dual_obj = solver.dual_obj_t
                elseif solver.goal_duinf < 1
                    verbose && println("dual infeasibility detected; terminating")
                    solver.status = DualInfeasible
                    solver.primal_obj = solver.primal_obj_t
                    solver.dual_obj = solver.dual_obj_t
                elseif solver.goal_ip < 1
                    verbose && println("ill-posedness detected; terminating")
                    solver.status = IllPosed
                end
                break
            end

            if impr < T(0.999)
                # update best point
                copyto!(solver.best_point.vec, point.vec)
            end

            if impr > 1 - solver.tol_slow
                # println("insufficient improvement: $impr")
                solver.slow_count += 1
                if solver.slow_count > 4 # TODO option
                    verbose && println("slow progress encountered; terminating")
                    # TODO reset to best point and force centering step?
                    # @warn("resetting to previous best point with goal $")
                    # TODO but have to go straight to start of next iter? not conv check below
                    solver.status = SlowProgress
                    break
                end
            else
                solver.slow_count = 0
            end

            if solver.num_iters == solver.iter_limit
                verbose && println("iteration limit reached; terminating")
                solver.status = IterationLimit
                break
            end
            if time() - start_time >= solver.time_limit
                verbose && println("time limit reached; terminating")
                solver.status = TimeLimit
                break
            end

            step(stepper, solver) || break
            flush(stdout)
            calc_mu(solver)

            if point.tau[] <= zero(T) || point.kap[] <= zero(T) || solver.mu <= zero(T)
                @warn("numerical failure: tau is $(point.tau[]), kappa is $(point.kap[]), mu is $(solver.mu); terminating")
                solver.status = NumericalFailure
                break
            end

            flush(stdout)
            solver.num_iters += 1
        end

        # finalize result point
        postprocess(solver)
    end

    solver.solve_time = time() - start_time

    # free memory used by some system solvers
    free_memory(solver.system_solver)

    verbose && println("\nstatus is $(solver.status) after $(solver.num_iters) iterations and $(trunc(solver.solve_time, digits=3)) seconds\n")
    flush(stdout)

    return solver
end

function calc_mu(solver::Solver{T}) where {T <: Real}
    point = solver.point
    solver.mu = (dot(point.z, point.s) + point.tau[] * point.kap[]) / (solver.model.nu + 1)
    return solver.mu
end

function calc_residuals(solver::Solver{T}) where {T <: Real}
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

    # y_residual = A*x - b*tau
    y_residual = solver.y_residual
    mul!(y_residual, model.A, point.x)
    solver.y_norm_res_t = norm(y_residual, Inf)
    @. y_residual -= model.b * tau
    solver.y_norm_res = norm(y_residual, Inf) / tau

    # z_residual = s + G*x - h*tau
    z_residual = solver.z_residual
    mul!(z_residual, model.G, point.x)
    @. z_residual += point.s
    solver.z_norm_res_t = norm(z_residual, Inf)
    @. z_residual -= model.h * tau
    solver.z_norm_res = norm(z_residual, Inf) / tau

    # tau_residual = c'*x + b'*y + h'*z + kap
    solver.primal_obj_t = dot(model.c, point.x)
    solver.dual_obj_t = -dot(model.b, point.y) - dot(model.h, point.z)
    solver.tau_residual = solver.primal_obj_t - solver.dual_obj_t + point.kap[]

    # auxiliary constraint and objective measures for printing and progress checks
    solver.primal_obj = solver.primal_obj_t / tau + model.obj_offset
    solver.dual_obj = solver.dual_obj_t / tau + model.obj_offset
    solver.gap = dot(point.z, point.s)
    solver.x_feas = solver.x_norm_res * solver.x_conv_tol
    solver.y_feas = solver.y_norm_res * solver.y_conv_tol
    solver.z_feas = solver.z_norm_res * solver.z_conv_tol
    solver.worst_feas = max(solver.x_feas, solver.y_feas, solver.z_feas, abs(solver.tau_residual))

    return nothing
end

function update_goals(solver::Solver{T}) where {T <: Real}
    tau = solver.point.tau[]
    primal_obj_t = solver.primal_obj_t
    dual_obj_t = solver.dual_obj_t

    # optimality conditions
    goal_feas = max(solver.x_feas, solver.y_feas, solver.z_feas) / solver.tol_feas
    goal_abs_opt = solver.gap / solver.tol_abs_opt
    denom_rel_opt = solver.tol_rel_opt * max(tau, min(abs(primal_obj_t), abs(dual_obj_t)))
    goal_rel_opt = min(solver.gap / tau, abs(primal_obj_t - dual_obj_t)) / denom_rel_opt
    goal_opt = max(goal_feas, min(goal_abs_opt, goal_rel_opt))

    # primal infeasibility conditions
    if dual_obj_t < eps(T)
        goal_prinf = T(Inf)
    else
        goal_prinf = solver.x_norm_res_t / (solver.tol_infeas * dual_obj_t)
    end

    # dual infeasibility conditions
    if primal_obj_t > -eps(T)
        goal_duinf = T(Inf)
    else
        goal_duinf = max(solver.y_norm_res_t, solver.z_norm_res_t) / (solver.tol_infeas * -primal_obj_t)
    end

    # ill-posed conditions # TODO experiment with these
    goal_ip = max(solver.mu / solver.tol_infeas, tau / (solver.tol_infeas * min(one(T), solver.point.kap[])))

    # summarize goals
    solver.goal = min(goal_opt, goal_prinf, goal_duinf, goal_ip)
    @assert solver.goal >= 0

    # get best goal improvement
    impr = T(Inf)
    for (a, b) in ((goal_opt, solver.goal_opt), (goal_prinf, solver.goal_prinf), (goal_duinf, solver.goal_duinf), (goal_ip, solver.goal_ip))
        impr_ab = a / b
        if !isnan(impr_ab)
            impr = min(impr, impr_ab)
        end
    end

    solver.goal_opt = goal_opt
    solver.goal_prinf = goal_prinf
    solver.goal_duinf = goal_duinf
    solver.goal_ip = goal_ip

    return impr
end

function initialize_cone_point(model::Models.Model{T}) where {T <: Real}
    init_z = zeros(T, model.q)
    init_s = zeros(T, model.q)

    for (cone, idxs) in zip(model.cones, model.cone_idxs)
        Cones.setup_data(cone)
        primal_k = view(Cones.use_dual_barrier(cone) ? init_z : init_s, idxs)
        dual_k = view(Cones.use_dual_barrier(cone) ? init_s : init_z, idxs)
        Cones.set_initial_point(primal_k, cone)
        Cones.load_point(cone, primal_k)
        @assert Cones.is_feas(cone) # TODO error?
        g = Cones.grad(cone)
        @. dual_k = -g
        Cones.load_dual_point(cone, dual_k)
        hasfield(typeof(cone), :hess_fact_cache) && @assert Cones.update_hess_fact(cone) # TODO error?
    end

    return (init_z, init_s)
end

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

function load(solver::Solver{T}, model::Models.Model{T}) where {T <: Real}
    # @assert solver.status == NotLoaded # TODO maybe want a reset function that just keeps options
    solver.orig_model = model
    solver.status = Loaded
    return solver
end

include("process.jl")

include("search.jl")

include("steppers/common.jl")

include("systemsolvers/common.jl")

# release memory used by sparse system solvers
free_memory(::SystemSolver) = nothing
free_memory(system_solver::Union{NaiveSparseSystemSolver, SymIndefSparseSystemSolver}) = free_memory(system_solver.fact_cache)

# function print_header(stepper::Stepper, solver::Solver)
#     @printf("\n%5s %9s %12s %12s %9s %9s ", "iter", "goal", "p_obj", "d_obj", "abs_gap", "x_feas")
#     if !iszero(solver.model.p)
#         @printf("%9s ", "y_feas")
#     end
#     @printf("%9s %9s %9s %9s ", "z_feas", "tau", "kap", "mu")
#     print_header_more(stepper, solver)
#     println()
#     return
# end
function print_header(stepper::Stepper, solver::Solver)
    println()
    @printf("%5s %12s %12s %9s ", "iter", "p_obj", "d_obj", "abs_gap")
    @printf("%9s %9s %9s %9s ", "feas", "tau", "kap", "mu")
    @printf("%9s %9s %9s %9s ", "opt", "p_inf", "d_inf", "ill_p")
    print_header_more(stepper, solver)
    println()
    return
end
print_header_more(stepper::Stepper, solver::Solver) = nothing

# function print_iteration(stepper::Stepper, solver::Solver)
#     @printf("%5d %9.2e %12.4e %12.4e %9.2e %9.2e ",
#         solver.num_iters, solver.goal, solver.primal_obj, solver.dual_obj, solver.gap, solver.x_feas
#         )
#     if !iszero(solver.model.p)
#         @printf("%9.2e ", solver.y_feas)
#     end
#     @printf("%9.2e %9.2e %9.2e %9.2e ",
#         solver.z_feas, solver.point.tau[], solver.point.kap[], solver.mu
#         )
#     print_iteration_more(stepper, solver)
#     println()
#     return
# end
function print_iteration(stepper::Stepper, solver::Solver)
    @printf("%5d %12.4e %12.4e %9.2e ",
        solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap)
    @printf("%9.2e %9.2e %9.2e %9.2e ",
        solver.worst_feas, solver.point.tau[], solver.point.kap[], solver.mu)
    @printf("%9.2e %9.2e %9.2e %9.2e ",
        solver.goal_opt, solver.goal_prinf, solver.goal_duinf, solver.goal_ip)
    print_iteration_more(stepper, solver)
    println()
    return
end
print_iteration_more(stepper::Stepper, solver::Solver) = nothing

end
