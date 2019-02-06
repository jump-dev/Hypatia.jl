



function combined_predict_correct(solver::HSDSolver)

    model = solver.model
    cones = model.cones
    point = solver.point
    (n, p, q) = (model.n, model.p, model.q)

    LHS = [
        zeros(n,n)  model.A'    model.G'          zeros(n)  zeros(n,q)         model.c;
        -model.A    zeros(p,p)  zeros(p,q)        zeros(p)  zeros(p,q)         model.b;
        zeros(q,n)  zeros(q,p)  Matrix(1.0I,q,q)  zeros(q)  Matrix(1.0I,q,q)   zeros(q);
        zeros(1,n)  zeros(1,p)  zeros(1,q)        1.0       zeros(1,q)         1.0;
        -model.G    zeros(q,p)  zeros(q,q)        zeros(q)  Matrix(-1.0I,q,q)  model.h;
        -model.c'   -model.b'   -model.h'         -1.0      zeros(1,q)         0.0;
    ]

    LHS[n+p+q+1, end] = solver.mu / solver.tau / solver.tau
    for k in eachindex(cones)
        cone_k = cones[k]
        # TODO stepped to this point so should already have called check_in_cone for the point
        Cones.load_point(cone_k, point.primal_views[k])
        @assert Cones.check_in_cone(cone_k)
        rows = (n + p) .+ model.cone_idxs[k]
        cols = Cones.use_dual(cone_k) ? rows : (q + 1) .+ rows
        LHS[rows, cols] = solver.mu * Cones.hess(cone_k)
    end

    rhs = [
        solver.x_residual  zeros(n);
        solver.y_residual  zeros(p);
        zeros(q)    zeros(q);
        -solver.kap  (-solver.kap + solver.mu / solver.tau);
        solver.z_residual  zeros(q);
        (solver.kap + solver.primal_obj_t - solver.dual_obj_t)  0.0;
        ]
    for k in eachindex(cones)
        rows = (n + p) .+ model.cone_idxs[k]
        rhs[rows, 1] = -point.dual_views[k]
        rhs[rows, 2] = -point.dual_views[k] - solver.mu * Cones.grad(cones[k])
    end

    F = lu(LHS)
    ldiv!(F, rhs)

    # affine phase
    # affine_direction = construct_affine_direction(direction_solution, mu, solver)
    # @views begin
        x_dirs = rhs[1:n, :]
        y_dirs = rhs[(n + 1):(n + p), :]
        z_dirs = rhs[(n + p + 1):(n + p + q), :]
        s_dirs = rhs[(n + p + q + 2):(n + p + 2q + 1), :]
        kap_dirs = rhs[n + p + q + 1, :]
        tau_dirs = rhs[n + p + 2q + 2, :]
    # end


    # # calculate prediction and correction directions
    # (x_dirs, y_dirs, z_dirs, s_dirs, tau_dirs, kap_dirs) = get_combined_directions(solver)

    # affine_alpha = find_max_alpha_in_nbhd(view(z_dirs, :, 1), view(s_dirs, :, 1), tau_dirs[1], kap_dirs[1], 0.99, solver)
    affine_alpha = find_max_alpha_in_nbhd(z_dirs[:, 1], s_dirs[:, 1], tau_dirs[1], kap_dirs[1], 0.999, solver)
    gamma = (1.0 - affine_alpha)^3 # TODO allow different function (heuristic)
    @show gamma

    comb_scaling = vcat(1.0 - gamma, gamma)
    z_comb = z_dirs * comb_scaling
    s_comb = s_dirs * comb_scaling
    tau_comb = dot(tau_dirs, comb_scaling)
    kap_comb = dot(kap_dirs, comb_scaling)
    alpha = find_max_alpha_in_nbhd(z_comb, s_comb, tau_comb, kap_comb, solver.max_nbhd, solver)
    @show alpha

    if iszero(alpha)
        alpha = 0.999
        gamma = 1.0
        comb_scaling = vcat(1.0 - gamma, gamma)

        z_comb = z_dirs * comb_scaling
        s_comb = s_dirs * comb_scaling
        tau_comb = dot(tau_dirs, comb_scaling)
        kap_comb = dot(kap_dirs, comb_scaling)
    end

    # point = solver.point
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





# function combined_predict_correct(point::Models.Point, residual::Models.Point, mu::Float64, solver::HSDSolver)
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
#         point.kap + primal_obj_t - dual_obj_t  0.0;
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
