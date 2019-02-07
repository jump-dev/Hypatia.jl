
mutable struct CombinedNaiveStepper <: HSDStepper
    lhs
    lhs_H_k
    rhs
    sol
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

    function CombinedNaiveStepper(model::Models.Linear)
        (n, p, q) = (model.n, model.p, model.q)
        stepper = new()

        # TODO allow sparse lhs
        stepper.lhs = [
            zeros(n,n)  model.A'    model.G'          zeros(n)  zeros(n,q)         model.c;
            -model.A    zeros(p,p)  zeros(p,q)        zeros(p)  zeros(p,q)         model.b;
            zeros(q,n)  zeros(q,p)  Matrix(1.0I,q,q)  zeros(q)  Matrix(1.0I,q,q)   zeros(q);
            zeros(1,n)  zeros(1,p)  zeros(1,q)        1.0       zeros(1,q)         1.0;
            -model.G    zeros(q,p)  zeros(q,q)        zeros(q)  Matrix(-1.0I,q,q)  model.h;
            -model.c'   -model.b'   -model.h'         -1.0      zeros(1,q)         0.0;
        ]
        function view_k(k::Int)
            rows = (n + p) .+ model.cone_idxs[k]
            cols = Cones.use_dual(model.cones[k]) ? rows : (q + 1) .+ rows
            return view(stepper.lhs, rows, cols)
        end
        stepper.lhs_H_k = [view_k(k) for k in eachindex(model.cones)]

        stepper.rhs = zeros(size(stepper.lhs, 1), 2)
        stepper.sol = similar(stepper.rhs)
        rows = 1:n
        stepper.x_rhs1 = view(stepper.rhs, rows, 1)
        stepper.x_sol = view(stepper.sol, rows, :)
        rows = (n + 1):(n + p)
        stepper.y_rhs1 = view(stepper.rhs, rows, 1)
        stepper.y_sol = view(stepper.sol, rows, :)
        stepper.z_rhs1_k = [view(stepper.rhs, (n + p) .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
        stepper.z_rhs2_k = [view(stepper.rhs, (n + p) .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]
        stepper.z_sol = view(stepper.sol, (n + p + 1):(n + p + q), :)
        stepper.kap_sol = view(stepper.sol, n + p + q + 1, :)
        rows = (n + p + q + 2):(n + p + 2q + 1)
        stepper.s_rhs1 = view(stepper.rhs, rows, 1)
        stepper.s_sol = view(stepper.sol, rows, :)
        stepper.tau_sol = view(stepper.sol, n + p + 2q + 2, :)

        return stepper
    end
end

function get_combined_directions(solver::HSDSolver, stepper::CombinedNaiveStepper)
    model = solver.model
    cones = model.cones
    point = solver.point
    lhs = stepper.lhs
    rhs = stepper.rhs
    (n, p, q) = (model.n, model.p, model.q)

    # update lhs matrix
    lhs[n + p + q + 1, end] = solver.mu / solver.tau / solver.tau
    for k in eachindex(cones)
        stepper.lhs_H_k[k] .= solver.mu * Cones.hess(cones[k])
    end

    # update rhs matrix
    row = n + p + q + 1
    rhs[row, 1] = -solver.kap
    rhs[row, 2] = -solver.kap + solver.mu / solver.tau
    rhs[row + q + 1, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    stepper.x_rhs1 .= solver.x_residual
    stepper.y_rhs1 .= solver.y_residual
    stepper.s_rhs1 .= solver.z_residual
    for k in eachindex(cones)
        stepper.z_rhs1_k[k] .= -point.dual_views[k]
        stepper.z_rhs2_k[k] .= -point.dual_views[k] - solver.mu * Cones.grad(cones[k])
    end

    # solve linear system
    F = lu(lhs) # TODO copy matrix and do in-place
    ldiv!(stepper.sol, F, rhs)

    return (stepper.x_sol, stepper.y_sol, stepper.z_sol, stepper.s_sol, stepper.tau_sol, stepper.kap_sol)
end
