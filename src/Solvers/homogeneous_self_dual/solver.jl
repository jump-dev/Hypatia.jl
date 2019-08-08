#=
Copyright 2018, Chris Coey and contributors

interior point type and functions for algorithms based on homogeneous self dual embedding
=#

mutable struct HSDSolver{T <: HypReal} <: Solver{T}
    model::Models.LinearModel{T}

    stepper::HSDStepper{T}

    verbose::Bool
    tol_rel_opt::T
    tol_abs_opt::T
    tol_feas::T
    tol_slow::T
    max_iters::Int
    time_limit::Float64

    x_conv_tol::T
    y_conv_tol::T
    z_conv_tol::T

    point::Models.Point{T}
    tau::T
    kap::T
    mu::T

    x_residual::Vector{T}
    y_residual::Vector{T}
    z_residual::Vector{T}
    x_norm_res_t::T
    y_norm_res_t::T
    z_norm_res_t::T
    x_norm_res::T
    y_norm_res::T
    z_norm_res::T

    primal_obj_t::T
    dual_obj_t::T
    primal_obj::T
    dual_obj::T
    gap::T
    rel_gap::T
    x_feas::T
    y_feas::T
    z_feas::T

    prev_is_slow::Bool
    prev2_is_slow::Bool
    prev_gap::T
    prev_rel_gap::T
    prev_x_feas::T
    prev_y_feas::T
    prev_z_feas::T

    status::Symbol
    num_iters::Int
    solve_time::Float64
    timer::TimerOutput

    function HSDSolver{T}(
        model::Models.LinearModel{T};
        stepper::HSDStepper{T} = CombinedHSDStepper{T}(model),
        verbose::Bool = true,
        tol_rel_opt = max(1e-12, 1e-2 * cbrt(eps(T))),
        tol_abs_opt = tol_rel_opt,
        tol_feas = tol_rel_opt,
        tol_slow = 5e-3,
        max_iters::Int = 250,
        time_limit = 3e2,
        ) where {T <: HypReal}
        solver = new{T}()

        solver.model = model

        solver.stepper = stepper

        solver.verbose = verbose
        solver.tol_rel_opt = T(tol_rel_opt)
        solver.tol_abs_opt = T(tol_abs_opt)
        solver.tol_feas = T(tol_feas)
        solver.tol_slow = T(tol_slow)
        solver.max_iters = max_iters
        solver.time_limit = time_limit

        solver.x_conv_tol = inv(max(one(T), norm(model.c)))
        solver.y_conv_tol = inv(max(one(T), norm(model.b)))
        solver.z_conv_tol = inv(max(one(T), norm(model.h)))

        solver.point = model.initial_point
        solver.tau = one(T)
        solver.kap = one(T)
        solver.mu = T(NaN)

        solver.x_residual = similar(model.c)
        solver.y_residual = similar(model.b)
        solver.z_residual = similar(model.h)
        solver.x_norm_res_t = T(NaN)
        solver.y_norm_res_t = T(NaN)
        solver.z_norm_res_t = T(NaN)
        solver.x_norm_res = T(NaN)
        solver.y_norm_res = T(NaN)
        solver.z_norm_res = T(NaN)

        solver.primal_obj_t = T(NaN)
        solver.dual_obj_t = T(NaN)
        solver.primal_obj = T(NaN)
        solver.dual_obj = T(NaN)
        solver.gap = T(NaN)
        solver.rel_gap = T(NaN)
        solver.x_feas = T(NaN)
        solver.y_feas = T(NaN)
        solver.z_feas = T(NaN)

        solver.prev_is_slow = false
        solver.prev2_is_slow = false
        solver.prev_gap = NaN
        solver.prev_rel_gap = NaN
        solver.prev_x_feas = NaN
        solver.prev_y_feas = NaN
        solver.prev_z_feas = NaN

        solver.status = :SolveNotCalled
        solver.num_iters = 0
        solver.solve_time = NaN
        solver.timer = TimerOutput()

        return solver
    end
end

get_tau(solver::HSDSolver) = solver.tau
get_kappa(solver::HSDSolver) = solver.kap
get_mu(solver::HSDSolver) = solver.mu

# TODO maybe use iteration interface rather than while loop
function solve(solver::HSDSolver{T}) where {T <: HypReal}
    solver.status = :SolveCalled
    start_time = time()

    calc_mu(solver)
    if isnan(solver.mu) || abs(one(T) - solver.mu) > sqrt(eps(T))
        error("initial mu is $(solver.mu) (should be 1)")
    end
    Cones.load_point.(solver.model.cones, solver.point.primal_views)

    while true
        @timeit solver.timer "calc_res" calc_residual(solver)

        @timeit solver.timer "calc_conv" calc_convergence_params(solver)

        @timeit solver.timer "print_iter" solver.verbose && print_iteration_stats(solver, solver.stepper)

        @timeit solver.timer "check_conv" check_convergence(solver) && break

        if solver.num_iters == solver.max_iters
            solver.verbose && println("iteration limit reached; terminating")
            solver.status = :IterationLimit
            break
        end

        if time() - start_time >= solver.time_limit
            solver.verbose && println("time limit reached; terminating")
            solver.status = :TimeLimit
            break
        end

        @timeit solver.timer "step" step(solver, solver.stepper)

        solver.num_iters += 1
    end

    # calculate result and iteration statistics and finish
    point = solver.point
    point.x ./= solver.tau
    point.y ./= solver.tau
    point.z ./= solver.tau
    point.s ./= solver.tau

    solver.solve_time = time() - start_time

    solver.verbose && println("\nstatus is $(solver.status) after $(solver.num_iters) iterations and $(trunc(solver.solve_time, digits=3)) seconds\n")

    return
end

function calc_mu(solver::HSDSolver{T}) where {T <: HypReal}
    solver.mu = (dot(solver.point.z, solver.point.s) + solver.tau * solver.kap) /
        (one(T) + solver.model.nu)
    return solver.mu
end

# A'*y + G'*z + c*tau = xrhs = (solver.x_residual, 0)
# -A*x + b*tau = yrhs = (solver.y_residual, 0)
# -G*x - s + h*tau = zrhs = (-duals_k, -duals_k - mu * g)
# -c'*x - b'*y - h'*z - kap = kaprhs = (-kap, -kap + mu / tau)
# (pr bar) z_k + mu*H_k*s_k = srhs_k = (z_residual, 0)
# (du bar) mu*H_k*z_k + s_k = srhs_k
# kap + mu/(taubar^2)*tau = taurhs = (kap + solver.primal_obj_t - solver.dual_obj_t, 0)

function calc_residuals_curr(solver::HSDSolver{T}, x_pred, x_corr, y_pred, y_corr, z_pred, z_corr, s_pred, s_corr, tau_pred, tau_corr, kap_pred, kap_corr) where {T <: HypReal}
    model = solver.model
    point = solver.point

    x_res_pred = solver.x_residual - model.A' * y_pred - model.G' * z_pred - model.c * tau_pred
    x_res_corr = - model.A' * y_corr - model.G' * z_corr - model.c * tau_corr

    y_res_pred = solver.y_residual + model.A * x_pred - model.b * tau_pred
    y_res_corr =  model.A * x_corr - model.b * tau_corr

    z_rhs_pred = []
    z_rhs_corr = []
    for k in eachindex(model.cones)
        duals_k = solver.point.dual_views[k]
        g = Cones.grad(model.cones[k])
        push!(z_rhs_pred, -duals_k...)
        push!(z_rhs_corr, (-duals_k - solver.mu * g)...)
    end
    z_res_pred = z_rhs_pred + model.G * x_pred + s_pred - model.h * tau_pred
    z_res_corr = z_rhs_corr + model.G * x_corr + s_corr - model.h * tau_corr

    kap_res_pred = -solver.kap + model.c' * x_pred + model.b' * y_pred + model.h' * z_pred + kap_pred
    kap_res_corr = -solver.kap + solver.mu / solver.tau + model.c' * x_corr + model.b' * y_corr + model.h' * z_corr + kap_corr

    s_res_pred = zeros(model.q)
    s_res_corr = zeros(model.q)
    for k in eachindex(model.cones)
        cone_k = model.cones[k]
        idxs = model.cone_idxs[k]
        if Cones.use_dual(cone_k)
            s_res_pred[idxs] = solver.z_residual[idxs] - solver.mu * Cones.hess(cone_k) * z_pred[idxs] - s_pred[idxs]
            s_res_corr[idxs] = -solver.mu * Cones.hess(cone_k) * z_corr[idxs] - s_corr[idxs]
        else
            s_res_pred[idxs] = solver.z_residual[idxs] - z_pred[idxs] - solver.mu * Cones.hess(cone_k) * s_pred[idxs]
            s_res_corr[idxs] = -z_corr[idxs] - solver.mu * Cones.hess(cone_k) * s_corr[idxs]
        end
    end

    tau_res_pred = solver.kap + solver.primal_obj_t - solver.dual_obj_t - kap_pred - solver.mu / solver.tau / solver.tau * tau_pred
    tau_res_corr = -kap_corr - solver.mu / solver.tau / solver.tau * tau_corr

    pred_res = vcat(x_res_pred, y_res_pred, z_res_pred, s_res_pred, kap_res_pred, tau_res_pred)
    corr_res = vcat(x_res_corr, y_res_corr, z_res_corr, s_res_corr, kap_res_corr, tau_res_corr)

    # @show norm(x_res_pred), norm(y_res_pred), norm(z_res_pred), norm(s_res_pred), norm(kap_res_pred), norm(tau_res_pred)

    return (pred_res, corr_res)
end

function calc_residual(solver::HSDSolver{T}) where {T <: HypReal}
    model = solver.model
    point = solver.point

    # x_residual = -A'*y - G'*z - c*tau
    x_residual = solver.x_residual
    mul!(x_residual, model.G', point.z)
    x_residual .= -model.A' * point.y - x_residual # TODO remove allocs
    solver.x_norm_res_t = norm(x_residual)
    @. x_residual -= model.c * solver.tau
    solver.x_norm_res = norm(x_residual) / solver.tau

    # y_residual = A*x - b*tau
    y_residual = solver.y_residual
    mul!(y_residual, model.A, point.x)
    solver.y_norm_res_t = norm(y_residual)
    @. y_residual -= model.b * solver.tau
    solver.y_norm_res = norm(y_residual) / solver.tau

    # z_residual = s + G*x - h*tau
    z_residual = solver.z_residual
    mul!(z_residual, model.G, point.x)
    @. z_residual += point.s
    solver.z_norm_res_t = norm(z_residual)
    @. z_residual -= model.h * solver.tau
    solver.z_norm_res = norm(z_residual) / solver.tau

    return
end

function calc_convergence_params(solver::HSDSolver{T}) where {T <: HypReal}
    model = solver.model
    point = solver.point

    solver.prev_gap = solver.gap
    solver.prev_rel_gap = solver.rel_gap
    solver.prev_x_feas = solver.x_feas
    solver.prev_y_feas = solver.y_feas
    solver.prev_z_feas = solver.z_feas

    solver.primal_obj_t = dot(model.c, point.x)
    solver.dual_obj_t = -dot(model.b, point.y) - dot(model.h, point.z)
    solver.primal_obj = solver.primal_obj_t / solver.tau
    solver.dual_obj = solver.dual_obj_t / solver.tau
    solver.gap = dot(point.z, point.s)
    if solver.primal_obj < zero(T)
        solver.rel_gap = solver.gap / -solver.primal_obj
    elseif solver.dual_obj > zero(T)
        solver.rel_gap = solver.gap / solver.dual_obj
    else
        solver.rel_gap = T(NaN)
    end

    solver.x_feas = solver.x_norm_res * solver.x_conv_tol
    solver.y_feas = solver.y_norm_res * solver.y_conv_tol
    solver.z_feas = solver.z_norm_res * solver.z_conv_tol

    return
end

function check_convergence(solver::HSDSolver{T}) where {T <: HypReal}
    # check convergence criteria
    # TODO nearly primal or dual infeasible or nearly optimal cases?
    if max(solver.x_feas, solver.y_feas, solver.z_feas) <= solver.tol_feas &&
        (solver.gap <= solver.tol_abs_opt || (!isnan(solver.rel_gap) && solver.rel_gap <= solver.tol_rel_opt))
        solver.verbose && println("optimal solution found; terminating")
        solver.status = :Optimal
        return true
    end
    if solver.dual_obj_t > zero(T)
        infres_pr = solver.x_norm_res_t * solver.x_conv_tol / solver.dual_obj_t
        if infres_pr <= solver.tol_feas
            solver.verbose && println("primal infeasibility detected; terminating")
            solver.status = :PrimalInfeasible
            return true
        end
    end
    if solver.primal_obj_t < zero(T)
        infres_du = -max(solver.y_norm_res_t * solver.y_conv_tol, solver.z_norm_res_t * solver.z_conv_tol) / solver.primal_obj_t
        if infres_du <= solver.tol_feas
            solver.verbose && println("dual infeasibility detected; terminating")
            solver.status = :DualInfeasible
            return true
        end
    end
    if solver.mu <= solver.tol_feas * T(1e-2) && solver.tau <= solver.tol_feas * T(1e-2) * min(one(T), solver.kap)
        solver.verbose && println("ill-posedness detected; terminating")
        solver.status = :IllPosed
        return true
    end

    max_improve = zero(T)
    for (curr, prev) in ((solver.gap, solver.prev_gap), (solver.rel_gap, solver.prev_rel_gap),
        (solver.x_feas, solver.prev_x_feas), (solver.y_feas, solver.prev_y_feas), (solver.z_feas, solver.prev_z_feas))
        if isnan(prev) || isnan(curr)
            continue
        end
        max_improve = max(max_improve, (prev - curr) / (abs(prev) + eps(T)))
    end
    if max_improve < solver.tol_slow
        if solver.prev_is_slow && solver.prev2_is_slow
            solver.verbose && println("slow progress in consecutive iterations; terminating")
            solver.status = :SlowProgress
            return true
        else
            solver.prev2_is_slow = solver.prev_is_slow
            solver.prev_is_slow = true
        end
    else
        solver.prev2_is_slow = solver.prev_is_slow
        solver.prev_is_slow = false
    end

    return false
end
