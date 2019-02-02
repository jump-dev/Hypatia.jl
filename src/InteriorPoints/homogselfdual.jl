#=
Copyright 2018, Chris Coey and contributors

interior point type and functions for algorithms based on homogeneous self dual embedding

TODO make internal statuses types
=#

mutable struct HSDESolver <: IPMSolver
    model::Models.LinearObjConic
    linear_solver::LinearSystems.LinearSystemSolver

    # options
    verbose::Bool
    tol_rel_opt::Float64
    tol_abs_opt::Float64
    tol_feas::Float64
    max_iters::Int
    time_limit::Float64
    combined_nbhd::Float64

    x_conv_tol::Float64
    y_conv_tol::Float64
    z_conv_tol::Float64

    # point
    point::Models.Point
    tau::Float64
    kap::Float64
    mu::Float64

    x_residual::Vector{Float64}
    y_residual::Vector{Float64}
    z_residual::Vector{Float64}
    x_norm_res_t::Float64
    y_norm_res_t::Float64
    z_norm_res_t::Float64
    x_norm_res::Float64
    y_norm_res::Float64
    z_norm_res::Float64

    obj_primal_t::Float64
    obj_dual_t::Float64
    obj_primal::Float64
    obj_dual::Float64
    gap::Float64
    rel_gap::Float64

    # solve info
    status::Symbol
    num_iters::Int
    solve_time::Float64
    primal_obj::Float64
    dual_obj::Float64

    function HSDESolver(
        model::Models.LinearObjConic,
        ;
        linear_solver_type::Type{<:LinearSystems.LinearSystemSolver} = LinearSystems.CholChol,
        verbose::Bool = true,
        tol_rel_opt = 1e-6,
        tol_abs_opt = 1e-7,
        tol_feas = 1e-7,
        max_iters::Int = 100,
        time_limit::Float64 = 3e2,
        combined_nbhd::Float64 = 0.1,
        )
        solver = new()

        solver.model = model
        solver.linear_solver = linear_solver_type(model)

        solver.verbose = verbose
        solver.tol_rel_opt = tol_rel_opt
        solver.tol_abs_opt = tol_abs_opt
        solver.tol_feas = tol_feas
        solver.max_iters = max_iters
        solver.time_limit = time_limit
        solver.combined_nbhd = combined_nbhd

        solver.x_conv_tol = inv(max(1.0, norm(model.c)))
        solver.y_conv_tol = inv(max(1.0, norm(model.b)))
        solver.z_conv_tol = inv(max(1.0, norm(model.h)))

        solver.point = find_initial_point(solver)
        solver.tau = 1.0
        solver.kap = 1.0
        solver.mu = NaN

        solver.x_residual = similar(model.c)
        solver.y_residual = similar(model.b)
        solver.z_residual = similar(model.h)
        solver.x_norm_res_t = NaN
        solver.y_norm_res_t = NaN
        solver.z_norm_res_t = NaN
        solver.x_norm_res = NaN
        solver.y_norm_res = NaN
        solver.z_norm_res = NaN

        solver.obj_primal_t = NaN
        solver.obj_dual_t = NaN
        solver.obj_primal = NaN
        solver.obj_dual = NaN
        solver.gap = NaN
        solver.rel_gap = NaN

        solver.status = :SolveNotCalled
        solver.num_iters = 0
        solver.solve_time = NaN
        solver.primal_obj = NaN
        solver.dual_obj = NaN

        return solver
    end
end

get_tau(solver::HSDESolver) = solver.tau
get_kappa(solver::HSDESolver) = solver.kap
get_mu(solver::HSDESolver) = solver.mu

# TODO maybe use iteration interface rather than while loop
function solve(solver::HSDESolver)
    solver.status = :SolveCalled
    start_time = time()

    calc_mu(solver)
    if isnan(solver.mu) || abs(1.0 - solver.mu) > 1e-6
        error("initial mu is $(solver.mu) (should be 1.0)")
    end

    solver.verbose && @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s\n",
        "iter", "p_obj", "d_obj", "abs_gap", "rel_gap", "p_inf", "d_inf", "tau", "kap", "mu")

    while true
        calc_residual(solver)

        if check_convergence(solver)
            break
        end

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

        # TODO may use different function, or function could change during some iteration eg if numerical difficulties
        combined_predict_correct(solver)

        solver.num_iters += 1
    end

    # calculate result and iteration statistics and finish
    point = solver.point
    point.x ./= solver.tau
    point.y ./= solver.tau
    point.z ./= solver.tau
    point.s ./= solver.tau

    solver.solve_time = time() - start_time

    if solver.verbose
        println("\nstatus is $(solver.status) after $(solver.num_iters) iterations and $(trunc(solver.solve_time, digits=3)) seconds\n")
    end

    return
end

function check_convergence(solver::HSDESolver)
    model = solver.model
    point = solver.point

    norm_res_primal = max(solver.y_norm_res * solver.y_conv_tol, solver.z_norm_res * solver.z_conv_tol)
    norm_res_dual = solver.x_norm_res * solver.x_conv_tol

    solver.obj_primal_t = dot(model.c, point.x)
    solver.obj_dual_t = -dot(model.b, point.y) - dot(model.h, point.z)
    solver.obj_primal = solver.obj_primal_t / solver.tau
    solver.obj_dual = solver.obj_dual_t / solver.tau
    solver.gap = dot(point.z, point.s) # TODO maybe should adapt original Alfonso condition instead of using this CVXOPT condition
    if solver.obj_primal < 0.0
        solver.rel_gap = solver.gap / -solver.obj_primal
    elseif solver.obj_dual > 0.0
        solver.rel_gap = solver.gap / solver.obj_dual
    else
        solver.rel_gap = NaN
    end

    # print iteration statistics
    if solver.verbose
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            solver.num_iters, solver.obj_primal, solver.obj_dual, solver.gap, solver.rel_gap,
            norm_res_primal, norm_res_dual, solver.tau, solver.kap, solver.mu
            )
        flush(stdout)
    end

    # check convergence criteria
    # TODO nearly primal or dual infeasible or nearly optimal cases?
    if norm_res_primal <= solver.tol_feas && norm_res_dual <= solver.tol_feas &&
        (solver.gap <= solver.tol_abs_opt || (!isnan(solver.rel_gap) && solver.rel_gap <= solver.tol_rel_opt))
        solver.verbose && println("optimal solution found; terminating")
        solver.status = :Optimal
        return true
    end
    if solver.obj_dual_t > 0.0
        infres_pr = solver.x_norm_res_t * solver.x_conv_tol / solver.obj_dual_t
        if infres_pr <= solver.tol_feas
            solver.verbose && println("primal infeasibility detected; terminating")
            solver.status = :PrimalInfeasible
            return true
        end
    end
    if solver.obj_primal_t < 0.0
        infres_du = -max(solver.y_norm_res_t * solver.y_conv_tol, solver.z_norm_res_t * solver.z_conv_tol) / solver.obj_primal_t
        if infres_du <= solver.tol_feas
            solver.verbose && println("dual infeasibility detected; terminating")
            solver.status = :DualInfeasible
            return true
        end
    end
    if solver.mu <= solver.tol_feas * 1e-2 && solver.tau <= solver.tol_feas * 1e-2 * min(1.0, solver.kap)
        solver.verbose && println("ill-posedness detected; terminating")
        solver.status = :IllPosed
        return true
    end

    return false
end

function combined_predict_correct(solver::HSDESolver)
    # calculate prediction and correction directions
    (x_dirs, y_dirs, z_dirs, s_dirs, tau_dirs, kap_dirs) = get_combined_directions(solver)

    affine_alpha = find_max_alpha_in_nbhd(view(z_dirs, :, 1), view(s_dirs, :, 1), tau_dirs[1], kap_dirs[1], 0.99, solver)
    gamma = (1.0 - affine_alpha)^3 # TODO allow different function (heuristic)
    # @show gamma

    comb_scaling = vcat(1.0 - gamma, gamma)
    z_comb = z_dirs * comb_scaling
    s_comb = s_dirs * comb_scaling
    tau_comb = (tau_dirs * comb_scaling)[1]
    kap_comb = (kap_dirs * comb_scaling)[1]
    alpha = find_max_alpha_in_nbhd(z_comb, s_comb, tau_comb, kap_comb, solver.combined_nbhd, solver)

    point = solver.point
    x_comb = x_dirs * comb_scaling
    y_comb = y_dirs * comb_scaling
    @. point.x += alpha * x_comb
    @. point.y += alpha * y_comb
    @. point.z += alpha * z_comb
    @. point.s += alpha * s_comb
    solver.tau += alpha * tau_comb
    solver.kap += alpha * kap_comb
    calc_mu(solver)

    return point
end

function get_combined_directions(solver::HSDESolver)
    model = solver.model
    cones = model.cones
    cone_idxs = model.cone_idxs
    mu = solver.mu

    # 2 columns:
    # 1) predictor/affine rhs
    # 2) corrector rhs
    # TODO prealloc, also note first col and some of 3rd col don't change
    x_rhss = hcat(solver.x_residual, zeros(model.n))
    y_rhss = hcat(solver.y_residual, zeros(model.p))
    z_rhss = zeros(model.q, 2)
    for k in eachindex(cones)
        idxs = model.cone_idxs[k]
        z_rhss[idxs, 1] = -point.dual_views[k]
        z_rhss[idxs, 2] = -point.dual_views[k] - mu * Cones.grad(cones[k])
    end
    s_rhss = hcat(solver.z_residual, zeros(model.q))
    kap_rhss = hcat(-solver.kap, -solver.kap + mu / solver.tau)
    tau_rhss = hcat(solver.kap + solver.obj_primal_t - solver.obj_dual_t, 0.0)

    (x_dirs, y_dirs, z_dirs, s_dirs, tau_dirs, kap_dirs) = LinearSystems.solve_linear_system(x_rhss, y_rhss, z_rhss, s_rhss, kap_rhss, tau_rhss, solver.linear_solver, solver)

    # for k in eachindex(cones)
    #     cone_k = cones[k]
    #     idxs = cone_idxs[k]
    #
    #     # first column
    #     h_k = view(model.h, idxs)
    #     if cone_k.use_dual
    #         z_rhs[idxs, 1] = Cones.inv_hess(cone_k) * (h_k ./ mu)
    #     else
    #         z_rhs[idxs, 1] = Cones.hess(cone_k) * (h_k .* mu)
    #     end
    #
    #     # second column
    #     z_k = solver.point.dual_views[k]
    #     s_k = solver.z_residual[idxs]
    #     if cone_k.use_dual
    #         z_rhs[idxs, 2] = Cones.inv_hess(cone_k) * ((z_k - s_k) ./ mu)
    #     else
    #         z_rhs[idxs, 2] = z_k - Cones.hess(cone_k) * (s_k .* mu)
    #     end
    #
    #     # third column
    #     z_k = solver.point.dual_views[k] + (Cones.grad(cones[k]) .* mu)
    #     if cone_k.use_dual
    #         z_rhs[idxs, 3] = Cones.inv_hess(cone_k) * (z_k ./ mu)
    #     else
    #         # z_rhs[idxs, 3] = Cones.hess(cone_k) * (z_k .* mu)
    #         z_rhs[idxs, 3] = z_k
    #     end
    # end

    # # call 3x3 solve routine
    # (x_sol, y_sol, z_sol) = LinearSystems.solve_linear_system(x_rhs, y_rhs, z_rhs, mu, model, solver.linear_solver)
    #
    # x1 = view(x_sol, :, 1)
    # y1 = view(y_sol, :, 1)
    # z1 = view(z_sol, :, 1)
    # x23 = view(x_sol, :, 2:3)
    # y23 = view(y_sol, :, 2:3)
    # z23 = view(z_sol, :, 2:3)
    #
    # # reconstruct using matrix operations
    # tau_rhs = [solver.kap + solver.obj_primal_t - solver.obj_dual_t  0.0]
    # kap_rhs = [-solver.kap  -solver.kap + mu / solver.tau]
    # tau_dirs_num = tau_rhs + kap_rhs + model.c' * x23 + model.b' * y23 + model.h' * z23
    # tau_dirs_den = mu / solver.tau / solver.tau - dot(model.c, x1) - dot(model.b, y1) - dot(model.h, z1)
    # tau_dirs = tau_dirs_num ./ tau_dirs_den
    #
    # x_dirs = x23 + x1 * tau_dirs
    # y_dirs = y23 + y1 * tau_dirs
    # z_dirs = z23 + z1 * tau_dirs
    #
    # s_dirs = -model.G * x_dirs + model.h * tau_dirs - [solver.z_residual  zeros(model.q)]
    # kap_dirs = -model.c' * x_dirs - model.b' * y_dirs - model.h' * z_dirs - tau_rhs

    return (x_dirs, y_dirs, z_dirs, s_dirs, tau_dirs, kap_dirs)
end

function calc_residual(solver::HSDESolver)
    model = solver.model
    point = solver.point

    # x_residual = -A'*y - G'*z - c*tau
    x_residual = solver.x_residual
    x_residual .= -model.A' * point.y - model.G' * point.z # TODO remove allocs
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

function calc_mu(solver::HSDESolver)
    solver.mu = (dot(solver.point.z, solver.point.s) + solver.tau * solver.kap) /
        (1.0 + solver.model.nu)
    return solver.mu
end

function find_initial_point(solver::HSDESolver)
    model = solver.model
    cones = model.cones
    point = Models.Point(model)

    for k in eachindex(cones)
        cone_k = cones[k]
        primal_k = point.primal_views[k]
        Cones.set_initial_point(primal_k, cone_k)
        Cones.load_point(cone_k, primal_k)
        @assert Cones.check_in_cone(cone_k)
        point.dual_views[k] .= -Cones.grad(cone_k)
    end

    # solve for y
    # A'y = -c - G'z
    # solve for x
    # Ax = b
    # Gx = h - s
    temp_n = -model.c - model.G' * point.z # TODO remove allocs
    temp_p_q = vcat(model.b, model.h - point.s) # TODO remove allocs
    # if has_QR(model) # TODO reuse QR from preprocessing
    #     ldiv!(point.y, model.At_qr, temp_n)
    #     ldiv!(point.x, model.AG_qr, temp_p_q)
    # else
        if !isempty(point.y)
            At_fact = factorize(issparse(model.A) ? sparse(model.A') : Matrix(model.A'))
            # ldiv!(point.y, At_fact, temp_n)
            point.y .= At_fact \ temp_n
        end
        AG_fact = factorize(vcat(model.A, model.G))
        # ldiv!(point.x, AG_fact, temp_p_q)
        point.x .= AG_fact \ temp_p_q
    # end

    return point
end

function find_max_alpha_in_nbhd(z_dir::AbstractVector{Float64}, s_dir::AbstractVector{Float64}, tau_dir::Float64, kap_dir::Float64, nbhd::Float64, solver::HSDESolver)
    point = solver.point
    model = solver.model
    cones = model.cones

    alpha = 1.0 # TODO maybe start at previous alpha but increased slightly, or use affine_alpha
    if kap_dir < 0.0
        alpha = min(alpha, -solver.kap / kap_dir)
    end
    if tau_dir < 0.0
        alpha = min(alpha, -solver.tau / tau_dir)
    end
    # TODO what about mu? quadratic equation. need dot(ls_s, ls_z) + ls_tau * ls_kap > 0
    alpha *= 0.99

    # ONLY BACKTRACK, NO FORWARDTRACK

    ls_z = similar(point.z) # TODO prealloc
    ls_s = similar(point.s)
    primal_views = [view(Cones.use_dual(cones[k]) ? ls_z : ls_s, model.cone_idxs[k]) for k in eachindex(cones)]
    dual_views = [view(Cones.use_dual(cones[k]) ? ls_s : ls_z, model.cone_idxs[k]) for k in eachindex(cones)]

    # cones_outside_nbhd = trues(length(cones))
    # TODO sort cones so that check the ones that failed in-cone check last iteration first

    ls_tau = ls_kap = ls_tk = ls_mu = 0.0
    num_pred_iters = 0
    while num_pred_iters < 100
        num_pred_iters += 1

        @. ls_z = point.z + alpha * z_dir
        @. ls_s = point.s + alpha * s_dir
        ls_tau = solver.tau + alpha * tau_dir
        ls_kap = solver.kap + alpha * kap_dir
        ls_tk = ls_tau * ls_kap
        ls_mu = (dot(ls_s, ls_z) + ls_tk) / (1.0 + model.nu)

        # accept primal iterate if
        # - decreased alpha and it is the first inside the cone and beta-neighborhood or
        # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
        # if ls_mu > 0.0 && abs(ls_tk - ls_mu) / ls_mu < nbhd # condition for 1-dim nonneg cone for tau and kap
        #     in_nbhds = true
        #     for k in eachindex(cones)
        #         cone_k = cones[k]
        #         Cones.load_point(cone_k, primal_views[k])
        #         if !Cones.check_in_cone(cone_k) || calc_neighborhood(cone_k, dual_views[k], ls_mu) > nbhd
        #             in_nbhds = false
        #             break
        #         end
        #     end
        #     if in_nbhds
        #         break
        #     end
        # end
        if ls_mu > 0.0
            full_nbhd_sqr = abs2(ls_tk - ls_mu)
            in_nbhds = true
            for k in eachindex(cones)
                cone_k = cones[k]
                Cones.load_point(cone_k, primal_views[k])
                if !Cones.check_in_cone(cone_k)
                    in_nbhds = false
                    break
                end

                # TODO no allocs
                temp = dual_views[k] + ls_mu * Cones.grad(cone_k)
                # TODO use cholesky L
                # nbhd = sqrt(temp' * Cones.inv_hess(cone) * temp) / mu
                full_nbhd_sqr += temp' * Cones.inv_hess(cone_k) * temp

                if full_nbhd_sqr > abs2(ls_mu * nbhd)
                    in_nbhds = false
                    break
                end
            end
            if in_nbhds
                break
            end
        end

        # iterate is outside the neighborhood: decrease alpha
        alpha *= 0.8 # TODO option for parameter
    end

    if alpha < 1e-7 # TODO return slow progress status or find a workaround
        error("alpha is $alpha")
    end

    return alpha
end




# function calc_neighborhood(cone::Cones.Cone, duals::AbstractVector{Float64}, mu::Float64)
#     # TODO no allocs
#     temp = duals + mu * Cones.grad(cone)
#     # TODO use cholesky L
#     # nbhd = sqrt(temp' * Cones.inv_hess(cone) * temp) / mu
#     nbhd = temp' * Cones.inv_hess(cone) * temp
#     # @show nbhd
#     return nbhd
# end


# function get_prediction_direction(point::Models.Point, residual::Models.Point, solver::HSDESolver)
#     for k in eachindex(solver.model.cones)
#         @. residual.tz_views[k] = -point.primal_views[k]
#     end
#     residual.tau = kap + cx + by + hz
#     residual.kap = -kap
#     return LinearSystems.solvelinsys6!(point, residual, mu, solver)
# end

# function get_correction_direction(point::Models.Point, solver::HSDESolver)
#     corr_rhs = solver.aux_point
#     @. corr_rhs.tx = 0.0
#     @. corr_rhs.ty = 0.0
#     cones = solver.model.cones
#     for k in eachindex(cones)
#         @. corr_rhs.tz_views[k] = -point.primal_views[k] - mu * Cones.grad(cones[k])
#     end
#     @. corr_rhs.ts = 0.0
#     corr_rhs.tau = 0.0
#     corr_rhs.kap = -point.kap + mu / point.tau
#     return LinearSystems.solvelinsys6!(point, corr_rhs, mu, solver)
# end

# function predict_then_correct(point::Models.Point, residual::Models.Point, mu::Float64, solver::HSDESolver)
#     cones = solver.model.cones
#
#     # prediction phase
#     direction = get_prediction_direction(point, residual, solver)
#     alpha = 0.9999 * get_max_alpha(point, direction, solver)
#     point = step_in_direction(point, direction, alpha)
#
#     # correction phase
#     num_corr_steps = 0
#     while nbhd > eta && num_corr_steps < solver.max_corr_steps
#         direction = get_correction_direction(point, solver)
#         point = step_in_direction(point, direction, 0.9999)
#         for k in eachindex(cones)
#             @assert Cones.check_in_cone(cones[k], point.primal_views[k])
#         end
#         num_corr_steps += 1
#     end
#     if num_corr_steps == solver.max_corr_steps
#         solver.verbose && println("corrector phase finished outside the eta-neighborhood; terminating")
#         solver.status = :CorrectorFail
#         return (false, point)
#     end
#
#     return (true, point)
# end





#
# # get neighborhood parameters depending on magnitude of barrier parameter and maximum number of correction steps
# # TODO calculate values from the formulae given in Papp & Yildiz "On A Homogeneous Interior-Point Algorithm for Non-Symmetric Convex Conic Optimization"
# function getbetaeta(maxcorrsteps::Int, bnu::Float64)
#     if maxcorrsteps <= 2
#         if bnu < 10.0
#             return (0.1810, 0.0733, 0.0225)
#         elseif bnu < 100.0
#             return (0.2054, 0.0806, 0.0263)
#         else
#             return (0.2190, 0.0836, 0.0288)
#         end
#     elseif maxcorrsteps <= 4
#         if bnu < 10.0
#             return (0.2084, 0.0502, 0.0328)
#         elseif bnu < 100.0
#             return (0.2356, 0.0544, 0.0380)
#         else
#             return (0.2506, 0.0558, 0.0411)
#         end
#     else
#         if bnu < 10.0
#             return (0.2387, 0.0305, 0.0429)
#         elseif bnu < 100.0
#             return (0.2683, 0.0327, 0.0489)
#         else
#             return (0.2844, 0.0332, 0.0525)
#         end
#     end
# end


# function construct_direction(direction, x_fix, y_fix, z_fix, x_var, y_var, z_var, solver)
#     model = solver.model
#
#     direction.tau = (rhs_tau + rhs_kap + dot(L.c, xi[:, 2]) + dot(L.b, yi[:, 2]) + dot(L.h, z2)) /
#         (mu / tau / tau - dot(L.c, xi[:, 1]) - dot(L.b, yi[:, 1]) - dot(L.h, z1))
#     @. @views rhs_tx = xi[:, 2] + tau_dirs * xi[:, 1]
#     @. @views rhs_ty = yi[:, 2] + tau_dirs * yi[:, 1]
#     @. rhs_tz = z2 + tau_dirs * z1
#
#     mul!(z1, L.G, rhs_tx)
#     @. rhs_ts = -z1 + L.h * tau_dirs - rhs_ts
#     kap_dirs = -dot(L.c, rhs_tx) - dot(L.b, rhs_ty) - dot(L.h, rhs_tz) - rhs_tau
#
#     return direction
# end


# function step_in_direction(point::Models.Point, direction::Models.Point, alpha::Float64)
#     @. point.tx += alpha * direction.tx
#     @. point.ty += alpha * direction.ty
#     @. point.tz += alpha * direction.tz
#     @. point.ts += alpha * direction.ts
#     point.tau += alpha * direction.tau
#     point.kap += alpha * direction.kap
#     return point
# end

# function combined_predict_correct(point::Models.Point, residual::Models.Point, mu::Float64, solver::HSDESolver)
#     cones = solver.model.cones
#     (n, p, q) = (model.n, model.p, model.q)
#
#     # calculate prediction and correction directions
#
#     LHS[n+p+q+1, end] = mu / point.tau / point.tau
#     for k in eachindex(cones)
#         cone_k = cones[k]
#         # TODO stepped to this point so should already have called check_in_cone for the point
#         Cones.load_point(cone_k, point.primal_views[k])
#         @assert Cones.check_in_cone(cone_k)
#         rows = (n + p) .+ model.cone_idxs[k]
#         cols = Cones.use_dual(cone_k) ? rows : (q + 1) .+ rows
#         LHS[rows, cols] = mu * Cones.hess(cone_k)
#     end
#
#     rhs = [
#         x_residual  zeros(n);
#         y_residual  zeros(p);
#         zeros(q)    zeros(q);
#         -point.kap  -point.kap + mu / point.tau;
#         z_residual  zeros(q);
#         point.kap + obj_primal_t - obj_dual_t  0.0;
#         ]
#     for k in eachindex(cones)
#         rows = (n + p) .+ model.cone_idxs[k]
#         rhs[rows, 1] = -point.dual_views[k]
#         rhs[rows, 2] = -point.dual_views[k] - mu * Cones.grad(cones[k])
#     end
#
#     F = lu(LHS)
#     ldiv!(F, rhs)
#
#     # affine phase
#     # affine_direction = construct_affine_direction(direction_solution, mu, solver)
#     @. @views begin
#         predict.tx = rhs[1:n, 1]
#         predict.ty = rhs[(n + 1):(n + p), 1]
#         predict.tz = rhs[(n + p + 1):(n + p + q), 1]
#         predict.ts = rhs[(n + p + q + 2):(n + p + 2q + 1), 1]
#     end
#     predict.kap = rhs[n + p + q + 1, 1]
#     predict.tau = rhs[n + p + 2q + 2, 1]
#
#     # affine_alpha = get_max_alpha(point, predict, solver)
#     affine_alpha = find_max_alpha_in_nbhd(point, predict, mu, 0.99, solver)
#
#
#     # # NOTE step in corrector direction here: not in description of algorithms?
#     # @. @views begin
#     #     correct.tx = rhs[1:n, 2]
#     #     correct.ty = rhs[(n + 1):(n + p), 2]
#     #     correct.tz = rhs[(n + p + 1):(n + p + q), 2]
#     #     correct.ts = rhs[(n + p + q + 2):(n + p + 2q + 1), 2]
#     # end
#     # correct.kap = rhs[n + p + q + 1, 2]
#     # correct.tau = rhs[n + p + 2q + 2, 2]
#     #
#     # point = step_in_direction(point, correct, 1.0)
#     # mu = get_mu(point, model)
#
#
#     # combined phase
#     gamma = (1.0 - affine_alpha)^3 # TODO allow different function (heuristic)
#     # @show gamma
#
#     # direction = construct_combined_direction(direction_solution, mu, gamma, solver)
#     combined_rhs = rhs * vcat(1.0 - gamma, gamma)
#     combined = predict
#     @. @views begin
#         combined.tx = combined_rhs[1:n]
#         combined.ty = combined_rhs[(n + 1):(n + p)]
#         combined.tz = combined_rhs[(n + p + 1):(n + p + q)]
#         combined.ts = combined_rhs[(n + p + q + 2):(n + p + 2q + 1)]
#     end
#     combined.kap = combined_rhs[n + p + q + 1]
#     combined.tau = combined_rhs[n + p + 2q + 2]
#
#     alpha = find_max_alpha_in_nbhd(point, combined, mu, solver.combined_nbhd, solver)
#
#     point = step_in_direction(point, combined, alpha)
#     mu = get_mu(point, model)
#
#     return point
# end
