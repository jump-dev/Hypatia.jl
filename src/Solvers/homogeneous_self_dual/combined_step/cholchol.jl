
mutable struct CombinedCholCholStepper <: HSDStepper

end

function get_combined_directions(solver::HSDSolver)
    model = solver.model
    cones = model.cones
    cone_idxs = model.cone_idxs
    mu = solver.mu
    point = solver.point

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

    # (x_dirs, y_dirs, z_dirs, s_dirs, tau_dirs, kap_dirs) = LinearSystems.solve_linear_system(x_rhss, y_rhss, z_rhss, s_rhss, kap_rhss, tau_rhss, solver.linear_solver)

    (n, p, q) = (model.n, model.p, model.q)

    LHS = [
        zeros(n,n)  model.A'    model.G'          zeros(n)  zeros(n,q)         model.c;
        -model.A    zeros(p,p)  zeros(p,q)        zeros(p)  zeros(p,q)         model.b;
        zeros(q,n)  zeros(q,p)  Matrix(1.0I,q,q)  zeros(q)  Matrix(1.0I,q,q)   zeros(q);
        zeros(1,n)  zeros(1,p)  zeros(1,q)        1.0       zeros(1,q)         1.0;
        -model.G    zeros(q,p)  zeros(q,q)        zeros(q)  Matrix(-1.0I,q,q)  model.h;
        -model.c'   -model.b'   -model.h'         -1.0      zeros(1,q)         0.0;
        ]

    LHS[n+p+q+1, end] = mu / solver.tau / solver.tau
    for k in eachindex(cones)
        cone_k = cones[k]
        # TODO stepped to this point so should already have called check_in_cone for the point
        # Cones.load_point(cone_k, point.primal_views[k])
        # @assert Cones.check_in_cone(cone_k)
        rows = (n + p) .+ cone_idxs[k]
        cols = Cones.use_dual(cone_k) ? rows : (q + 1) .+ rows
        LHS[rows, cols] = mu * Cones.hess(cone_k)
    end

    rhs = vcat(x_rhss, y_rhss, z_rhss, kap_rhss, s_rhss, tau_rhss)

    sol = LHS \ rhs

    return (sol[1:n, :], sol[n+1:n+p, :], sol[n+p+1:n+p+q, :], sol[n+p+q+2:n+p+2q+1, :], sol[n+p+2q+2:n+p+2q+2, :], sol[n+p+q+1:n+p+q+1, :])

    #     @. @views begin
    #         predict.tx = rhs[1:n, 1]
    #         predict.ty = rhs[(n + 1):(n + p), 1]
    #         predict.tz = rhs[(n + p + 1):(n + p + q), 1]
    #         predict.ts = rhs[(n + p + q + 2):(n + p + 2q + 1), 1]
    #     end
    #     predict.kap = rhs[n + p + q + 1, 1]
    #     predict.tau = rhs[n + p + 2q + 2, 1]


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

    # return (x_dirs, y_dirs, z_dirs, s_dirs, tau_dirs, kap_dirs)
end
