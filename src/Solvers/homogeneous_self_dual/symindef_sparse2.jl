#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in symindef.jl

TODO CHOLMOD ldlt expects sparse matrix to have Int64 rowvals/colptrs while
pardiso wants Int32 -_- so do these differently to avoid allocs

serious TODO add epsilons on the diagonal, in particular for pardiso
=#

mutable struct SymIndefSparseSystemSolver <: SystemSolver{Float64}
    use_inv_hess::Bool

    tau_row
    lhs
    hess_idxs
    sparse_cache

    function SymIndefSparseSystemSolver(;
        # sparse_cache = PardisoCache(true)
        sparse_cache = CHOLMODCache()
        )
        system_solver = new()
        system_solver.sparse_cache = sparse_cache
        return system_solver
    end
end

function load(system_solver::SymIndefSparseSystemSolver, solver::Solver{Float64})
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    (A, G, b, h, c) = (model.A, model.G, model.b, model.h, model.c)
    system_solver.tau_row = n + p + q + 1
    # TODO remove
    A = sparse(A)
    G = sparse(G)
    dropzeros!(A)
    dropzeros!(G)

    # count the number of nonzeros we will have in the lhs
    hess_nnzs = sum(Cones.hess_nnzs(cone_k) for cone_k in model.cones)
    nnzs = nnz(A) + nnz(G) + hess_nnzs
    Is = Vector{Int64}(undef, nnzs)
    Js = Vector{Int64}(undef, nnzs)
    Vs = Vector{Float64}(undef, nnzs)

    # count of nonzeros added so far
    offset = 1
    # update I, J, V while adding A and G blocks to the lhs
    offset = Solvers.add_I_J_V(offset, Is, Js, Vs, [n, n + p], [0, 0], [A, G], [false, false])
    @timeit solver.timer "setup hess lhs" begin
    nz_rows_added = 0
    for (k, cone_k) in enumerate(model.cones)
        @timeit solver.timer "update hess" Cones.update_hess(cone_k)
        cone_dim = Cones.dimension(cone_k)
        rows = n + p + nz_rows_added
        offset = add_I_J_V(offset, Is, Js, Vs, rows, rows, cone_k, !Cones.use_dual(cone_k))
        nz_rows_added += cone_dim
    end
    end # hess timing
    @assert offset == nnzs + 1
    dim = n + p + q
    # NOTE only lower block-triangle was constructed
    @timeit solver.timer "build sparse" system_solver.lhs = sparse(Is, Js, Vs, Int64(dim), Int64(dim))
    lhs = system_solver.lhs

    # cache indices of placeholders of Hessians
    @timeit solver.timer "cache idxs" begin
    system_solver.hess_idxs = [Vector{UnitRange}(undef, Cones.dimension(cone_k)) for cone_k in model.cones]
    row = col = n + p + 1
    for (k, cone_k) in enumerate(model.cones)
        cone_dim = Cones.dimension(cone_k)
        for j in 1:cone_dim
            # get list of nonzero rows in the current column of the LHS
            col_idx_start = lhs.colptr[col]
            col_idx_end = lhs.colptr[col + 1] - 1
            nz_rows = lhs.rowval[col_idx_start:col_idx_end]
            # nonzero rows in column j of the hessian
            nz_hess_indices = Cones.hess_nz_idxs_j(cone_k, j)
            # index corresponding to first nonzero Hessian element of the current column of the LHS
            offset_in_row = findfirst(x -> x == row + nz_hess_indices[1] - 1, nz_rows)
            # indices of nonzero values for cone k column j
            system_solver.hess_idxs[k][j] = col_idx_start + offset_in_row - nz_hess_indices[1] .+ nz_hess_indices .- 1
            # move to the next column
            col += 1
        end
        row += cone_dim
    end
    end # cache timing


    # system_solver.lhs = T[
    #     spzeros(T,n,n)  spzeros(T,n,p)  spzeros(T,n,q);
    #     model.A         spzeros(T,p,p)  spzeros(T,p,q);
    #     model.G         spzeros(T,q,p)  sparse(-one(T)*I,q,q);
    #     ]

    return system_solver
end

# update the LHS factorization to prepare for solve
function update_fact(system_solver::SymIndefSparseSystemSolver, solver::Solver{Float64})
    reset_sparse_cache(system_solver.sparse_cache)

    cones = solver.model.cones
    @timeit solver.timer "modify views" begin
    for (k, cone_k) in enumerate(cones)
        @timeit solver.timer "update hess" H = (Cones.use_dual(cone_k) ? -Cones.hess(cone_k) : -Cones.inv_hess(cone_k))
        for j in 1:Cones.dimension(cone_k)
            nz_rows = Cones.hess_nz_idxs_j(cone_k, j)
            @views copyto!(system_solver.lhs.nzval[system_solver.hess_idxs[k][j]], H[nz_rows, j])
        end
    end
    end # time views

    model = solver.model
    n, p, q = model.n, model.p, model.q
    lhs_compare = [
        spzeros(n,n)  spzeros(n,p)  spzeros(n,q);
        model.A         spzeros(p,p)  spzeros(p,q);
        model.G         spzeros(q,p)  sparse(-I,q,q);
        ]
    rc = n + p + 1
    for (k, cone_k) in enumerate(cones)
        dim = Cones.dimension(cone_k)
        H = (Cones.use_dual(cone_k) ? -Cones.hess(cone_k) : -Cones.inv_hess(cone_k))
        lhs_compare[rc:(rc + dim - 1), rc:(rc + dim - 1)] .= H
        rc += dim
    end

    if norm(Symmetric(lhs_compare - system_solver.lhs, :L)) > 1
        println("###############################")
    end

    #
    # lhs_symm = Symmetric(lhs, :L)
    # if system_solver.use_sparse
    #     system_solver.fact_cache = ldlt(lhs_symm, shift = eps(T))
    # else
    #     system_solver.fact_cache = (T == BigFloat ? lu!(lhs_symm) : bunchkaufman!(lhs_symm))
    # end

    return system_solver
end

# solve system without outer iterative refinement
function solve_system(system_solver::SymIndefSparseSystemSolver, solver::Solver{Float64}, sol, rhs)
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = system_solver.tau_row
    cache = system_solver.sparse_cache
    lhs = system_solver.lhs

    # TODO in-place
    dim3 = tau_row - 1
    sol3 = zeros(dim3, 3)
    rhs3 = zeros(dim3, 3)

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
        else
            # G_k*x - (mu*H_k)\z_k = [-zrhs_k - (mu*H_k)\srhs_k, h_k]
            Cones.inv_hess_prod!(zk12_new, sk12, cone_k)
            @. zk12_new *= -1
            @. zk12_new -= zk12
            @. zk3_new = hk
        end
    end

    # if !cache.analyzed
    #     analyze_sparse_system(cache, lhs, rhs3)
    #     cache.analyzed = true
    # end
    # @timeit solver.timer "solve system" solve_sparse_system(cache, sol3, lhs, rhs3, solver)
    sol3 .= Symmetric(lhs, :L) \ rhs3

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
    mul!(s, model.G, sol[1:n, :], -one(Float64), true)
    @. @views s -= rhs[(n + p) .+ (1:q), :]

    # kap = -mu/(taubar^2)*tau + kaprhs
    @. @views sol[end:end, :] = -solver.mu / solver.tau * tau / solver.tau + rhs[end:end, :]

    return sol
end
