#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in symindef.jl
=#

mutable struct SymIndefSparseSystemSolver <: SystemSolver{Float64}
    use_inv_hess::Bool

    tau_row
    lhs
    sparse_cache

    function SymIndefSparseSystemSolver(;
        # sparse_cache = PardisoCache()
        sparse_cache = SuiteSparseCache()
        )
        system_solver = new()
        system_solver.sparse_cache = sparse_cache
        return system_solver
    end
end

function load(system_solver::SymIndefSparseSystemSolver, solver::Solver{Float64})
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    system_solver.tau_row = n + p + q + 1
    model = solver.model
    dropzeros!(model.A)
    dropzeros!(model.G)

    # fill symmetric lower triangle

    # system_solver.lhs = T[
    #     spzeros(T,n,n)  spzeros(T,n,p)  spzeros(T,n,q);
    #     model.A         spzeros(T,p,p)  spzeros(T,p,q);
    #     model.G         spzeros(T,q,p)  sparse(-one(T)*I,q,q);
    #     ]



    return system_solver
end

# update the LHS factorization to prepare for solve
function update_fact(system_solver::SymIndefSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    reset_sparse_cache(system_solver.sparse_cache)
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    lhs = system_solver.lhs

    if !system_solver.use_sparse
        copyto!(lhs, system_solver.lhs_copy)
    end

    lhs[end, end] = solver.mu / solver.tau / solver.tau
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        if Cones.use_dual(cone_k)
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            H = Cones.hess(cone_k)
            @. lhs[z_rows_k, z_rows_k] = -H
        else system_solver.use_inv_hess
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Hinv = Cones.inv_hess(cone_k)
            @. lhs[z_rows_k, z_rows_k] = -Hinv
        end

    end

    lhs_symm = Symmetric(lhs, :L)
    if system_solver.use_sparse
        system_solver.fact_cache = ldlt(lhs_symm, shift = eps(T))
    else
        system_solver.fact_cache = (T == BigFloat ? lu!(lhs_symm) : bunchkaufman!(lhs_symm))
    end

    return system_solver
end

# solve system without outer iterative refinement
function solve_system(system_solver::SymIndefSparseSystemSolver{T}, solver::Solver{T}, sol, rhs) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = system_solver.tau_row

    # TODO in-place
    dim3 = tau_row - 1
    sol3 = zeros(T, dim3, 3)
    rhs3 = zeros(T, dim3, 3)

    @. @views rhs3[1:n, 1:2] = rhs[1:n, :]
    @. @views rhs3[n .+ (1:p), 1:2] = -rhs[n .+ (1:p), :]
    @. rhs3[1:n, 3] = -model.c
    @. rhs3[n .+ (1:p), 3] = model.b

    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        z_rows_k = (n + p) .+ idxs_k
        s_rows_k = tau_row .+ idxs_k
        zk12 = @view rhs[z_rows_k, :]
        sk12 = @view rhs[s_rows_k, :]
        hk = @view model.h[idxs_k]
        zk12_new = @view rhs3[z_rows_k, 1:2]
        zk3_new = @view rhs3[z_rows_k, 3]

        if Cones.use_dual(cone_k)
            # G_k*x - mu*H_k*z_k = [-zrhs_k - srhs_k, h_k]
            @. zk12_new = -zk12 - sk12
            @. zk3_new = hk
        else system_solver.use_inv_hess
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Cones.inv_hess_prod!(zk12_new, sk12, cone_k)
            @. zk12_new *= -1
            @. zk12_new -= zk12
            @. zk3_new = hk
        end
    end

    if system_solver.use_iterative
        error("not implemented")
        # for j in 1:size(rhs3, 2)
        #     rhs_j = view(rhs3, :, j)
        #     sol_j = view(sol3, :, j)
        #     IterativeSolvers.minres!(sol_j, system_solver.lhs, rhs_j, restart = tau_row)
        # end
    else
        if system_solver.use_sparse
            sol3 .= system_solver.fact_cache \ rhs3
        else
            # if !hyp_bk_solve!(system_solver.fact_cache, sol3, lhs, rhs3)
            #     @warn("numerical failure: could not fix linear solve failure (mu is $(solver.mu))")
            # end
            ldiv!(sol3, system_solver.fact_cache, rhs3)
        end
    end

    if !system_solver.use_inv_hess
        for (k, cone_k) in enumerate(model.cones)
            if !Cones.use_dual(cone_k)
                # recover z_k = mu*H_k*w_k
                z_rows_k = (n + p) .+ model.cone_idxs[k]
                z_copy_k = sol3[z_rows_k, :] # TODO do in-place
                @views Cones.hess_prod!(sol3[z_rows_k, :], z_copy_k, cone_k)
            end
        end
    end

    x3 = @view sol3[1:n, 3]
    y3 = @view sol3[n .+ (1:p), 3]
    z3 = @view sol3[(n + p) .+ (1:q), 3]
    x12 = @view sol3[1:n, 1:2]
    y12 = @view sol3[n .+ (1:p), 1:2]
    z12 = @view sol3[(n + p) .+ (1:q), 1:2]

    # lift to get tau
    # TODO maybe use higher precision here
    tau_denom = solver.mu / solver.tau / solver.tau - dot(model.c, x3) - dot(model.b, y3) - dot(model.h, z3)
    tau = @view sol[tau_row:tau_row, :]
    @. @views tau = rhs[tau_row:tau_row, :] + rhs[end:end, :]
    tau .+= model.c' * x12 + model.b' * y12 + model.h' * z12 # TODO in place
    @. tau /= tau_denom

    @. x12 += tau * x3
    @. y12 += tau * y3
    @. z12 += tau * z3

    @views sol[1:dim3, :] = sol3[:, 1:2]

    # lift to get s and kap
    # TODO refactor below for use with symindef and qrchol methods
    # s = -G*x + h*tau - zrhs
    s = @view sol[(tau_row + 1):(end - 1), :]
    mul!(s, model.h, tau)
    mul!(s, model.G, sol[1:n, :], -one(T), true)
    @. @views s -= rhs[(n + p) .+ (1:q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs[end:end, :]

    return sol
end
