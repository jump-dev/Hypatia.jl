
mutable struct NaiveCombinedHSDSystemSolver <: CombinedHSDSystemSolver
    lhs_copy::Matrix{Float64}
    lhs::Matrix{Float64}
    lhs_H_k
    rhs::Matrix{Float64}
    x1
    x2
    y1
    y2
    z1
    z2
    z1_k
    z2_k
    s1
    s2
    kap_row::Int

    function NaiveCombinedHSDSystemSolver(model::Models.LinearModel)
        (n, p, q) = (model.n, model.p, model.q)
        system_solver = new()

        # TODO eliminate s and allow sparse lhs?
        # x y z kap s tau
        system_solver.lhs_copy = [
            zeros(n,n)  model.A'    model.G'          zeros(n)  zeros(n,q)         model.c;
            -model.A    zeros(p,p)  zeros(p,q)        zeros(p)  zeros(p,q)         model.b;
            zeros(q,n)  zeros(q,p)  Matrix(1.0I,q,q)  zeros(q)  Matrix(1.0I,q,q)   zeros(q);
            zeros(1,n)  zeros(1,p)  zeros(1,q)        1.0       zeros(1,q)         1.0;
            -model.G    zeros(q,p)  zeros(q,q)        zeros(q)  Matrix(-1.0I,q,q)  model.h;
            -model.c'   -model.b'   -model.h'         -1.0      zeros(1,q)         0.0;
        ]
        system_solver.lhs = similar(system_solver.lhs_copy)
        function view_k(k::Int)
            rows = (n + p) .+ model.cone_idxs[k]
            cols = Cones.use_dual(model.cones[k]) ? rows : (q + 1) .+ rows
            return view(system_solver.lhs, rows, cols)
        end
        system_solver.lhs_H_k = [view_k(k) for k in eachindex(model.cones)]

        system_solver.rhs = zeros(size(system_solver.lhs, 1), 2)
        rows = 1:n
        system_solver.x1 = view(system_solver.rhs, rows, 1)
        system_solver.x2 = view(system_solver.rhs, rows, 2)
        rows = (n + 1):(n + p)
        system_solver.y1 = view(system_solver.rhs, rows, 1)
        system_solver.y2 = view(system_solver.rhs, rows, 2)
        rows = (n + p + 1):(n + p + q)
        system_solver.z1 = view(system_solver.rhs, rows, 1)
        system_solver.z2 = view(system_solver.rhs, rows, 2)
        z_start = n + p
        system_solver.z1_k = [view(system_solver.rhs, z_start .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
        system_solver.z2_k = [view(system_solver.rhs, z_start .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]
        rows = (n + p + q + 2):(n + p + 2q + 1)
        system_solver.s1 = view(system_solver.rhs, rows, 1)
        system_solver.s2 = view(system_solver.rhs, rows, 2)
        system_solver.kap_row = n + p + q + 1

        return system_solver
    end
end

function get_combined_directions(solver::HSDSolver, system_solver::NaiveCombinedHSDSystemSolver)
    model = solver.model
    cones = model.cones
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    kap_row = system_solver.kap_row
    mu = solver.mu
    tau = solver.tau
    kap = solver.kap

    # update lhs matrix
    copyto!(lhs, system_solver.lhs_copy)
    lhs[kap_row, end] = mu / tau / tau
    for k in eachindex(cones)
        H = Cones.hess(cones[k])
        @. system_solver.lhs_H_k[k] = mu * H
    end

    # update rhs matrix
    system_solver.x1 .= solver.x_residual
    system_solver.x2 .= 0.0
    system_solver.y1 .= solver.y_residual
    system_solver.y2 .= 0.0
    for k in eachindex(cones)
        duals_k = solver.point.dual_views[k]
        g = Cones.grad(cones[k])
        @. system_solver.z1_k[k] = -duals_k
        @. system_solver.z2_k[k] = -duals_k - mu * g
    end
    system_solver.s1 .= solver.z_residual
    system_solver.s2 .= 0.0
    rhs[kap_row, 1] = -kap
    rhs[kap_row, 2] = -kap + mu / tau
    rhs[end, 1] = kap + solver.primal_obj_t - solver.dual_obj_t
    rhs[end, 2] = 0.0

    # solve system
    ldiv!(lu!(lhs), rhs)

    return (system_solver.x1, system_solver.x2, system_solver.y1, system_solver.y2, system_solver.z1, system_solver.z2, system_solver.s1, system_solver.s2, rhs[end, 1], rhs[end, 2], rhs[kap_row, 1], rhs[kap_row, 2])
end
