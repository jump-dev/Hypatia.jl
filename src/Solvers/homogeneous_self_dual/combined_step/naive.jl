
mutable struct NaiveCombinedHSDSystemSolver <: CombinedHSDSystemSolver
    lhs_copy::Matrix{Float64}
    lhs::Matrix{Float64}
    lhs_H_k
    rhs::Matrix{Float64}
    sol::Matrix{Float64}
    x_rhs1
    x_sol
    y_rhs1
    y_sol
    z_rhs1_k
    z_rhs2_k
    z_sol
    kap_sol
    s_rhs1
    s_rhs2
    s_sol
    tau_sol

    function NaiveCombinedHSDSystemSolver(model::Models.LinearModel)
        (n, p, q) = (model.n, model.p, model.q)
        system_solver = new()

        # TODO eliminate s and allow sparse lhs?
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
        system_solver.sol = similar(system_solver.rhs)
        rows = 1:n
        system_solver.x_rhs1 = view(system_solver.rhs, rows, 1)
        system_solver.x_sol = view(system_solver.sol, rows, :)
        rows = (n + 1):(n + p)
        system_solver.y_rhs1 = view(system_solver.rhs, rows, 1)
        system_solver.y_sol = view(system_solver.sol, rows, :)
        system_solver.z_rhs1_k = [view(system_solver.rhs, (n + p) .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
        system_solver.z_rhs2_k = [view(system_solver.rhs, (n + p) .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]
        system_solver.z_sol = view(system_solver.sol, (n + p + 1):(n + p + q), :)
        system_solver.kap_sol = view(system_solver.sol, n + p + q + 1, :)
        rows = (n + p + q + 2):(n + p + 2q + 1)
        system_solver.s_rhs1 = view(system_solver.rhs, rows, 1)
        system_solver.s_sol = view(system_solver.sol, rows, :)
        system_solver.tau_sol = view(system_solver.sol, n + p + 2q + 2, :)

        return system_solver
    end
end

function get_combined_directions(solver::HSDSolver, system_solver::NaiveCombinedHSDSystemSolver)
    model = solver.model
    cones = model.cones
    point = solver.point
    lhs = system_solver.lhs
    rhs = system_solver.rhs
    (n, p, q) = (model.n, model.p, model.q)

    # update lhs matrix
    copyto!(lhs, system_solver.lhs_copy)
    lhs[n + p + q + 1, end] = solver.mu / solver.tau / solver.tau
    for k in eachindex(cones)
        system_solver.lhs_H_k[k] .= solver.mu * Cones.hess(cones[k])
    end

    # update rhs matrix
    row = n + p + q + 1
    rhs[row, 1] = -solver.kap
    rhs[row, 2] = -solver.kap + solver.mu / solver.tau
    rhs[row + q + 1, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    system_solver.x_rhs1 .= solver.x_residual
    system_solver.y_rhs1 .= solver.y_residual
    system_solver.s_rhs1 .= solver.z_residual
    for k in eachindex(cones)
        system_solver.z_rhs1_k[k] .= -point.dual_views[k]
        system_solver.z_rhs2_k[k] .= -point.dual_views[k] - solver.mu * Cones.grad(cones[k])
    end

    # solve linear system_solver
    ldiv!(system_solver.sol, lu!(lhs), rhs)

    return (system_solver.x_sol, system_solver.y_sol, system_solver.z_sol, system_solver.s_sol, system_solver.tau_sol, system_solver.kap_sol)
end
