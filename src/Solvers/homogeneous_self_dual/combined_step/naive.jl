
mutable struct NaiveCombinedHSDSystemSolver{T <: HypReal} <: CombinedHSDSystemSolver{T}
    use_sparse::Bool

    lhs_copy
    lhs
    lhs_H_k
    rhs::Matrix{T}

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

    function NaiveCombinedHSDSystemSolver(model::Models.LinearModel{T}; use_sparse::Bool = false) where {T <: HypReal}
        (n, p, q) = (model.n, model.p, model.q)
        system_solver = new{T}()
        system_solver.use_sparse = use_sparse

        # x y z kap s tau
        if use_sparse
            system_solver.lhs_copy = T[
            spzeros(T,n,n)  model.A'        model.G'              spzeros(T,n)  spzeros(T,n,q)         model.c;
            -model.A        spzeros(T,p,p)  spzeros(T,p,q)        spzeros(T,p)  spzeros(T,p,q)         model.b;
            spzeros(T,q,n)  spzeros(T,q,p)  sparse(one(T)*I,q,q)  spzeros(T,q)  sparse(one(T)*I,q,q)   spzeros(T,q);
            spzeros(T,1,n)  spzeros(T,1,p)  spzeros(T,1,q)        one(T)        spzeros(T,1,q)         one(T);
            -model.G        spzeros(T,q,p)  spzeros(T,q,q)        spzeros(T,q)  sparse(-one(T)*I,q,q)  model.h;
            -model.c'       -model.b'       -model.h'             -one(T)       spzeros(T,1,q)         zero(T);
            ]
            @assert issparse(system_solver.lhs_copy)
        else
            system_solver.lhs_copy = T[
                zeros(T,n,n)  model.A'      model.G'              zeros(T,n)  zeros(T,n,q)           model.c;
                -model.A      zeros(T,p,p)  zeros(T,p,q)          zeros(T,p)  zeros(T,p,q)           model.b;
                zeros(T,q,n)  zeros(T,q,p)  Matrix(one(T)*I,q,q)  zeros(T,q)  Matrix(one(T)*I,q,q)   zeros(T,q);
                zeros(T,1,n)  zeros(T,1,p)  zeros(T,1,q)          one(T)      zeros(T,1,q)           one(T);
                -model.G      zeros(T,q,p)  zeros(T,q,q)          zeros(T,q)  Matrix(-one(T)*I,q,q)  model.h;
                -model.c'     -model.b'     -model.h'             -one(T)     zeros(T,1,q)           zero(T);
            ]
        end

        system_solver.lhs = similar(system_solver.lhs_copy)
        function view_k(k::Int)
            rows = (n + p) .+ model.cone_idxs[k]
            cols = Cones.use_dual(model.cones[k]) ? rows : (q + 1) .+ rows
            return view(system_solver.lhs, rows, cols)
        end
        system_solver.lhs_H_k = [view_k(k) for k in eachindex(model.cones)]

        system_solver.rhs = zeros(T, size(system_solver.lhs, 1), 2)
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

function get_combined_directions(solver::HSDSolver{T}, system_solver::NaiveCombinedHSDSystemSolver{T}) where T
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
    system_solver.x2 .= zero(T)
    system_solver.y1 .= solver.y_residual
    system_solver.y2 .= zero(T)
    for k in eachindex(cones)
        duals_k = solver.point.dual_views[k]
        g = Cones.grad(cones[k])
        @. system_solver.z1_k[k] = -duals_k
        @. system_solver.z2_k[k] = -duals_k - mu * g
    end
    system_solver.s1 .= solver.z_residual
    system_solver.s2 .= zero(T)
    rhs[kap_row, 1] = -kap
    rhs[kap_row, 2] = -kap + mu / tau
    rhs[end, 1] = kap + solver.primal_obj_t - solver.dual_obj_t
    rhs[end, 2] = zero(T)

    # solve system
    if system_solver.use_sparse
        rhs .= lu(lhs) \ rhs
    else
        ldiv!(lu!(lhs), rhs)
    end

    return (system_solver.x1, system_solver.x2, system_solver.y1, system_solver.y2, system_solver.z1, system_solver.z2, system_solver.s1, system_solver.s2, rhs[end, 1], rhs[end, 2], rhs[kap_row, 1], rhs[kap_row, 2])
end
