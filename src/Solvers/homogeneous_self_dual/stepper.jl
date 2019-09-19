#=
Copyright 2019, Chris Coey and contributors

interior point stepping routines for algorithms based on homogeneous self dual embedding
=#

# mutable struct CombinedStepper{T <: Real} <: Stepper{T}
#     solver::Solver{T}
#     use_iterative::Bool
#     use_sparse::Bool
#
#     rhs::Matrix{T}
#     rhs_x1
#     rhs_x2
#     rhs_y1
#     rhs_y2
#     rhs_z1
#     rhs_z2
#     rhs_s1
#     rhs_s2
#     rhs_s1_k
#     rhs_s2_k
#
#     sol::Matrix{T}
#     sol_x1
#     sol_x2
#     sol_y1
#     sol_y2
#     sol_z1
#     sol_z2
#     sol_s1
#     sol_s2
#
#     lhs_copy
#     lhs
#
#     fact_cache
#
# end

function step(solver::Solver{T}) where {T <: Real}
    model = solver.model
    point = solver.point

    # calculate affine/prediction and correction directions
    @timeit solver.timer "directions" (x_pred, x_corr, y_pred, y_corr, z_pred, z_corr, tau_pred, tau_corr, s_pred, s_corr, kap_pred, kap_corr) = get_combined_directions(solver.system_solver)

    # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    # TODO try setting nbhd to T(Inf) and avoiding the neighborhood checks - requires tuning
    @timeit solver.timer "aff_alpha" aff_alpha = find_max_alpha_in_nbhd(z_pred, s_pred, tau_pred, kap_pred, solver, nbhd = one(T), prev_alpha = max(solver.prev_aff_alpha, T(2e-2)), min_alpha = T(2e-2))
    solver.prev_aff_alpha = aff_alpha

    gamma = abs2(one(T) - aff_alpha) # TODO allow different function (heuristic)
    solver.prev_gamma = gamma

    # find distance alpha for stepping in combined direction
    z_comb = z_pred
    s_comb = s_pred
    pred_factor = one(T) - gamma
    @. z_comb = pred_factor * z_pred + gamma * z_corr
    @. s_comb = pred_factor * s_pred + gamma * s_corr
    tau_comb = pred_factor * tau_pred + gamma * tau_corr
    kap_comb = pred_factor * kap_pred + gamma * kap_corr
    @timeit solver.timer "comb_alpha" alpha = find_max_alpha_in_nbhd(z_comb, s_comb, tau_comb, kap_comb, solver, nbhd = solver.max_nbhd, prev_alpha = solver.prev_alpha, min_alpha = T(1e-2))

    if iszero(alpha)
        # could not step far in combined direction, so perform a pure correction step
        solver.verbose && println("performing correction step")
        z_comb = z_corr
        s_comb = s_corr
        tau_comb = tau_corr
        kap_comb = kap_corr
        @timeit solver.timer "corr_alpha" alpha = find_max_alpha_in_nbhd(z_comb, s_comb, tau_comb, kap_comb, solver, nbhd = solver.max_nbhd, prev_alpha = one(T), min_alpha = T(1e-6))

        if iszero(alpha)
            @warn("numerical failure: could not step in correction direction; terminating")
            solver.status = :NumericalFailure
            solver.keep_iterating = false
            return point
        end
        @. point.x += alpha * x_corr
        @. point.y += alpha * y_corr
    else
        @. point.x += alpha * (pred_factor * x_pred + gamma * x_corr)
        @. point.y += alpha * (pred_factor * y_pred + gamma * y_corr)
    end
    solver.prev_alpha = alpha

    # step distance alpha in combined direction
    @. point.z += alpha * z_comb
    @. point.s += alpha * s_comb
    solver.tau += alpha * tau_comb
    solver.kap += alpha * kap_comb
    calc_mu(solver)

    if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
        @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
        solver.status = :NumericalFailure
        solver.keep_iterating = false
    end

    return point
end

# return directions
# TODO make this function the same for all system solvers, move to solver.jl
# function get_combined_directions(system_solver::NaiveSystemSolver{T}) where {T <: Real}
function get_combined_directions(system_solver::SystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    rhs = system_solver.rhs
    sol = system_solver.sol

    update_rhs(system_solver)
    if !system_solver.use_iterative
        update_fact(system_solver)
    end
    solve_system(system_solver, sol, rhs) # NOTE dense solve with cache destroys RHS

    iter_ref_steps = 4 # TODO handle, maybe change dynamically
    for i in 1:iter_ref_steps
        # perform iterative refinement step
        res = calc_system_residual(solver, sol)
        norm_inf = norm(res, Inf)
        norm_2 = norm(res, 2)

        if norm_inf > eps(T)
            sol_curr = zeros(T, size(res, 1), 2)
            res_sol = solve_system(system_solver, sol_curr, res)
            sol_new = sol - res_sol
            res_new = calc_system_residual(solver, sol_new)
            norm_inf_new = norm(res_new, Inf)
            norm_2_new = norm(res_new, 2)
            if norm_inf_new < norm_inf && norm_2_new < norm_2
                solver.verbose && @printf("used iter ref, norms: inf %9.2e to %9.2e, two %9.2e to %9.2e\n", norm_inf, norm_inf_new, norm_2, norm_2_new)
                copyto!(sol, sol_new)
            else
                break
            end
        end
    end

    return (system_solver.sol_x1, system_solver.sol_x2, system_solver.sol_y1, system_solver.sol_y2, system_solver.sol_z1, system_solver.sol_z2, sol[system_solver.tau_row, 1], sol[system_solver.tau_row, 2], system_solver.sol_s1, system_solver.sol_s2, sol[end, 1], sol[end, 2])
end

# TODO experimental for block LHS: if block is a Cone then define mul as hessian product, if block is solver then define mul by mu/tau/tau
# TODO optimize... maybe need for each cone a 5-arg hess prod
import LinearAlgebra.mul!

function mul!(y::AbstractVecOrMat{T}, A::Cones.Cone{T}, x::AbstractVecOrMat{T}, alpha::Number, beta::Number) where {T <: Real}
    # TODO in-place
    ytemp = y * beta
    Cones.hess_prod!(y, x, A)
    rmul!(y, alpha)
    y .+= ytemp
    return y
end

function mul!(y::AbstractVecOrMat{T}, solver::Solvers.Solver{T}, x::AbstractVecOrMat{T}, alpha::Number, beta::Number) where {T <: Real}
    rmul!(y, beta)
    @. y += alpha * x / solver.tau * solver.mu / solver.tau
    return y
end

# update the 6x2 RHS matrix
function update_rhs(system_solver::SystemSolver{T}) where {T <: Real}
    solver = system_solver.solver

    system_solver.rhs_x1 .= solver.x_residual
    system_solver.rhs_x2 .= zero(T)
    system_solver.rhs_y1 .= solver.y_residual
    system_solver.rhs_y2 .= zero(T)
    system_solver.rhs_z1 .= solver.z_residual
    system_solver.rhs_z2 .= zero(T)
    system_solver.rhs[system_solver.tau_row, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    system_solver.rhs[system_solver.tau_row, 2] = zero(T)

    sqrtmu = sqrt(solver.mu)
    for (k, cone_k) in enumerate(solver.model.cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        @. system_solver.rhs_s1_k[k] = -duals_k
        @. system_solver.rhs_s2_k[k] = -duals_k - grad_k * sqrtmu
    end

    system_solver.rhs[end, 1] = -solver.kap
    system_solver.rhs[end, 2] = -solver.kap + solver.mu / solver.tau

    return system_solver.rhs
end

# calculate residual on 6x6 linear system
# TODO make efficient / in-place
function calc_system_residual(solver, sol)
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = n + p + q + 1

    @views begin
        sol_x = sol[1:n, :]
        sol_y = sol[(n + 1):(n + p), :]
        sol_z = sol[(n + p + 1):(n + p + q), :]
        sol_tau = sol[tau_row:tau_row, :]
        sol_s = sol[tau_row .+ (1:q), :]
        sol_kap = sol[end:end, :]
    end

    # A'*y + G'*z + c*tau = [x_residual, 0]
    res_x = model.A' * sol_y + model.G' * sol_z + model.c * sol_tau
    @. res_x[:, 1] -= solver.x_residual
    # -A*x + b*tau = [y_residual, 0]
    res_y = -model.A * sol_x + model.b * sol_tau
    @. res_y[:, 1] -= solver.y_residual
    # -G*x + h*tau - s = [z_residual, 0]
    res_z = -model.G * sol_x + model.h * sol_tau - sol_s
    @. res_z[:, 1] -= solver.z_residual
    # -c'*x - b'*y - h'*z - kap = [kap + primal_obj_t - dual_obj_t, 0]
    res_tau = -model.c' * sol_x - model.b' * sol_y - model.h' * sol_z - sol_kap
    res_tau[1] -= solver.kap + solver.primal_obj_t - solver.dual_obj_t

    sqrtmu = sqrt(solver.mu)
    res_s = similar(res_z)
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        if Cones.use_dual(cone_k)
            # (du bar) mu*H_k*z_k + s_k = srhs_k
            @views Cones.hess_prod!(res_s[idxs_k, :], sol_z[idxs_k, :], cone_k)
            @. @views res_s[idxs_k, :] += sol_s[idxs_k, :]
        else
            # (pr bar) z_k + mu*H_k*s_k = srhs_k
            @views Cones.hess_prod!(res_s[idxs_k, :], sol_s[idxs_k, :], cone_k)
            @. @views res_s[idxs_k, :] += sol_z[idxs_k, :]
        end
        # srhs_k = [-duals_k, -duals_k - mu * grad_k]
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cone_k)
        @. @views res_s[idxs_k, 1] += duals_k
        @. @views res_s[idxs_k, 2] += duals_k + grad_k * sqrtmu
    end

    # mu/(taubar^2)*tau + kap = [-kap, -kap + mu/tau]
    res_kap = sol_kap + solver.mu / solver.tau * sol_tau / solver.tau
    res_kap[1] += solver.kap
    res_kap[2] += solver.kap - solver.mu / solver.tau

    return vcat(res_x, res_y, res_z, res_tau, res_s, res_kap)
end

# backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
function find_max_alpha_in_nbhd(
    z_dir::AbstractVector{T},
    s_dir::AbstractVector{T},
    tau_dir::T,
    kap_dir::T,
    solver::Solver{T};
    nbhd::T,
    prev_alpha::T,
    min_alpha::T,
    ) where {T <: Real}
    point = solver.point
    model = solver.model
    z_temp = solver.z_temp
    s_temp = solver.s_temp

    alpha = min(prev_alpha * T(1.4), one(T)) # TODO option for parameter
    if kap_dir < zero(T)
        alpha = min(alpha, -solver.kap / kap_dir)
    end
    if tau_dir < zero(T)
        alpha = min(alpha, -solver.tau / tau_dir)
    end
    alpha *= T(0.9999)

    solver.cones_infeas .= true
    tau_temp = kap_temp = taukap_temp = mu_temp = zero(T)
    while true
        @timeit solver.timer "ls_update" begin
        @. z_temp = point.z + alpha * z_dir
        @. s_temp = point.s + alpha * s_dir
        tau_temp = solver.tau + alpha * tau_dir
        kap_temp = solver.kap + alpha * kap_dir
        taukap_temp = tau_temp * kap_temp
        mu_temp = (dot(s_temp, z_temp) + taukap_temp) / (one(T) + model.nu)
        end

        if mu_temp > zero(T)
            @timeit solver.timer "nbhd_check" in_nbhd = check_nbhd(mu_temp, taukap_temp, nbhd, solver)
            if in_nbhd
                break
            end
        end

        if alpha < min_alpha
            # alpha is very small so finish
            alpha = zero(T)
            break
        end

        # iterate is outside the neighborhood: decrease alpha
        alpha *= T(0.8) # TODO option for parameter
    end

    return alpha
end

function check_nbhd(
    mu_temp::T,
    taukap_temp::T,
    nbhd::T,
    solver::Solver{T},
    ) where {T <: Real}
    cones = solver.model.cones
    sqrtmu = sqrt(mu_temp)

    rhs_nbhd = mu_temp * abs2(nbhd)
    lhs_nbhd = abs2(taukap_temp / sqrtmu - sqrtmu)
    if lhs_nbhd >= rhs_nbhd
        return false
    end

    Cones.load_point.(cones, solver.primal_views, sqrtmu)

    # accept primal iterate if it is inside the cone and neighborhood
    # first check inside cone for whichever cones were violated last line search iteration
    for (k, cone_k) in enumerate(cones)
        if solver.cones_infeas[k]
            Cones.reset_data(cone_k)
            if Cones.is_feas(cone_k)
                solver.cones_infeas[k] = false
                solver.cones_loaded[k] = true
            else
                return false
            end
        else
            solver.cones_loaded[k] = false
        end
    end

    for (k, cone_k) in enumerate(cones)
        if !solver.cones_loaded[k]
            Cones.reset_data(cone_k)
            if !Cones.is_feas(cone_k)
                return false
            end
        end

        # modifies dual_views
        duals_k = solver.dual_views[k]
        g_k = Cones.grad(cone_k)
        @. duals_k += g_k * sqrtmu

        if solver.use_infty_nbhd
            k_nbhd = abs2(norm(duals_k, Inf) / norm(g_k, Inf))
            # k_nbhd = abs2(maximum(abs(dj) / abs(gj) for (dj, gj) in zip(duals_k, g_k))) # TODO try this neighborhood
            lhs_nbhd = max(lhs_nbhd, k_nbhd)
        else
            nbhd_temp_k = solver.nbhd_temp[k]
            Cones.inv_hess_prod!(nbhd_temp_k, duals_k, cone_k)
            k_nbhd = dot(duals_k, nbhd_temp_k)
            if k_nbhd <= -cbrt(eps(T))
                @warn("numerical failure: cone neighborhood is $k_nbhd")
                return false
            elseif k_nbhd > zero(T)
                lhs_nbhd += k_nbhd
            end
        end

        if lhs_nbhd > rhs_nbhd
            return false
        end
    end

    return true
end
