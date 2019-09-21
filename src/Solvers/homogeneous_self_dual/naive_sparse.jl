#=
Copyright 2018, Chris Coey and contributors

naive linear system solver

6x6 nonsymmetric system in (x, y, z, tau, s, kap):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
-G*x + h*tau - s = zrhs
-c'*x - b'*y - h'*z - kap = taurhs
(pr bar) z_k + mu*H_k*s_k = srhs_k
(du bar) mu*H_k*z_k + s_k = srhs_k
mu/(taubar^2)*tau + kap = kaprhs

TODO updates in Cones for epinorminf
=#

mutable struct NaiveSparseSystemSolver <: SparseSystemSolver
     tau_row
     lhs
     hess_idxs
     hess_view_k_j
     sparse_cache::SparseSolverCache
     solvecache
     mtt_idx::Int

     function NaiveSparseSystemSolver(;
         sparse_cache = PardisoCache(false)
         # sparse_cache = UMFPACKCache()
         )
         system_solver = new()
         system_solver.sparse_cache = sparse_cache
         return system_solver
     end
 end

# create the system_solver cache
function load(system_solver::NaiveSparseSystemSolver, solver::Solver{Float64})
    @timeit solver.timer "load" begin
    reset_sparse_cache(system_solver.sparse_cache)
    model = solver.model
    (A, G, b, h, c) = (model.A, model.G, model.b, model.h, model.c)
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs
    tau_row = n + p + q + 1
    system_solver.tau_row = tau_row
    dim = n + p + 2q + 2

    # TODO remove
    A = sparse(A)
    G = sparse(G)

    dropzeros!(A)
    dropzeros!(G)

    # x y z kap s tau

    # system_solver.lhs_actual_copy = T[
    #     spzeros(T,n,n)  model.A'        model.G'              model.c       spzeros(T,n,q)         spzeros(T,n);
    #     -model.A        spzeros(T,p,p)  spzeros(T,p,q)        model.b       spzeros(T,p,q)         spzeros(T,p);
    #     -model.G        spzeros(T,q,p)  spzeros(T,q,q)        model.h       sparse(-one(T)*I,q,q)  spzeros(T,q);
    #     -model.c'       -model.b'       -model.h'             zero(T)       spzeros(T,1,q)         -one(T);
    #     spzeros(T,q,n)  spzeros(T,q,p)  sparse(one(T)*I,q,q)  spzeros(T,q)  sparse(one(T)*I,q,q)   spzeros(T,q);
    #     spzeros(T,1,n)  spzeros(T,1,p)  spzeros(T,1,q)        one(T)        spzeros(T,1,q)         one(T);
    #     ]
    # dropzeros!(system_solver.lhs_actual_copy)
    # system_solver.lhs_actual = similar(system_solver.lhs_actual_copy)

    # count the number of nonzeros we will have in the lhs
    hess_nnzs = sum(Cones.dimension(cone_k) + Cones.hess_nnzs(cone_k) for cone_k in model.cones)
    nnzs = 2 * (nnz(A) + nnz(G) + n + p + q + 1) + q + 1 + hess_nnzs
    Is = Vector{Int32}(undef, nnzs)
    Js = Vector{Int32}(undef, nnzs)
    Vs = Vector{Float64}(undef, nnzs)

    # compute the sarting rows/columns for each known block in the lhs
    rc1 = 0
    rc2 = n
    rc3 = n + p
    rc4 = n + p + q
    rc5 = n + p + q + 1
    rc6 = dim - 1

    # count of nonzeros added so far
    offset = 1
    # add vectors in the lhs
    offset = add_I_J_V(
        offset, Is, Js, Vs,
        # start rows
        [rc1, rc2, rc3, fill(rc4, 3)..., rc4, rc6, rc6],
        # start cols
        [fill(rc4, 3)..., rc1, rc2, rc3, rc6, rc4, rc6],
        # vecs
        [c, b, h, -c, -b, -h, [-1.0], [1.0], [1.0]],
        # transpose
        vcat(fill(false, 3), fill(true, 3), fill(false, 3)),
        )
    # add sparse matrix blocks to the lhs
    offset = add_I_J_V(
        offset, Is, Js, Vs,
        # start rows
        [rc1, rc1, rc2, rc3, rc3],
        # start cols
        [rc2, rc3, rc1, rc1, rc5],
        # mats
        [A, G, -A, -G, sparse(-I, q, q)],
        # transpose
        [true, true, false, false, false],
        )

    # add I, J, V for Hessians
    @timeit solver.timer "setup hess lhs" begin
    nz_rows_added = 0
    for (k, cone_k) in enumerate(model.cones)
        cone_dim = Cones.dimension(cone_k)
        rows = rc5 + nz_rows_added
        dual_cols = rc3 + nz_rows_added
        is_dual = Cones.use_dual(cone_k)
        # add each Hessian's sparsity pattern in one placeholder block, an identity in the other
        H_cols = (is_dual ? dual_cols : rows)
        id_cols = (is_dual ? rows : dual_cols)
        offset = add_I_J_V(offset, Is, Js, Vs, rows, H_cols, cone_k, false)
        offset = add_I_J_V(offset, Is, Js, Vs, rows, id_cols, sparse(I, cone_dim, cone_dim), false)
        nz_rows_added += cone_dim
    end
    end # hess timing
    @assert offset == nnzs + 1

    @timeit solver.timer "build sparse" system_solver.lhs = sparse(Is, Js, Vs, Int32(dim), Int32(dim))
    lhs = system_solver.lhs

    # cache indices of placeholders of Hessians
    @timeit solver.timer "cache idxs" begin
    system_solver.hess_idxs = [Vector{UnitRange}(undef, Cones.dimension(cone_k)) for cone_k in model.cones]
    row = rc5 + 1
    col_offset = 1
    for (k, cone_k) in enumerate(model.cones)
        cone_dim = Cones.dimension(cone_k)
        init_col = (Cones.use_dual(cone_k) ? rc3 : rc5)
        for j in 1:cone_dim
            col = init_col + col_offset
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
            col_offset += 1
        end
        row += cone_dim
    end
    end # cache timing

    # get mtt index
    system_solver.mtt_idx = lhs.colptr[rc4 + 2] - 1

    # TODO currently not used, follow up what goes wrong here for soc cone
    system_solver.hess_view_k_j = [[view(cone_k.hess, :, j) for j in 1:Cones.dimension(cone_k)] for cone_k in model.cones]

    end # load timing

    return system_solver
end

# update the LHS factorization to prepare for solve
function update_fact(system_solver::NaiveSparseSystemSolver, solver::Solver{Float64})
    cones = solver.model.cones
    @timeit solver.timer "modify views" begin
    for (k, cone_k) in enumerate(cones)
        @timeit solver.timer "update hess" Cones.update_hess(cone_k)
        for j in 1:Cones.dimension(cone_k)
            nz_rows = Cones.hess_nz_idxs_j(cone_k, j)
            @views copyto!(system_solver.lhs.nzval[system_solver.hess_idxs[k][j]], cone_k.hess[nz_rows, j])
            # @views copyto!(system_solver.lhs.nzval[system_solver.hess_idxs[k][j]], system_solver.hess_view_k_j[k][j])
        end
    end
    end # time views
    mtt = solver.mu / solver.tau / solver.tau
    system_solver.lhs.nzval[system_solver.mtt_idx] = mtt
    return system_solver
end

function solve_system(system_solver::NaiveSparseSystemSolver, solver::Solver{Float64}, sol, rhs)
    cache = system_solver.sparse_cache
    lhs = system_solver.lhs
    if !cache.analyzed
        analyze_sparse_system(cache, lhs, rhs)
        cache.analyzed = true
    end
    @timeit solver.timer "solve system" solve_sparse_system(cache, sol, lhs, rhs, solver)
    return sol
end
