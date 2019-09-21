#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

naive linear system solver

6x6 nonsymmetric system in (x, y, z, tau, s, kap):
A'*y + G'*z + c*tau = xrhs
-A*x + b*tau = yrhs
-G*x + h*tau - s = zrhs
-c'*x - b'*y - h'*z - kap = taurhs
(pr bar) z_k + mu*H_k*s_k = srhs_k
(du bar) mu*H_k*z_k + s_k = srhs_k
mu/(taubar^2)*tau + kap = kaprhs

TODO for iterative method
- precondition
- optimize operations
- fix IterativeSolvers so that methods can take matrix RHS
=#

abstract type NaiveSystemSolver{T <: Real} <: SystemSolver{T} end

#=
indirect
=#

mutable struct NaiveIndirectSystemSolver{T <: Real} <: NaiveSystemSolver{T}
    lhs
    NaiveSystemSolver{T}() where {T <: Real} = new{T}()
end

function load(system_solver::NaiveIndirectSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs
    tau_row = n + p + q + 1
    system_solver.tau_row = tau_row
    dim = n + p + 2q + 2

    # setup block LHS
    rc1 = 1:n
    rc2 = n .+ (1:p)
    rc3 = (n + p) .+ (1:q)
    rc4 = tau_row:tau_row
    rc5 = tau_row .+ (1:q)
    rc6 = dim:dim

    k_len = 2 * length(model.cones)
    cone_rows = Vector{UnitRange{Int}}(undef, k_len)
    cone_cols = Vector{UnitRange{Int}}(undef, k_len)
    cone_blocks = Vector{Any}(undef, k_len)
    for (k, cone_k) in enumerate(model.cones)
        idxs_k = model.cone_idxs[k]
        rows = tau_row .+ idxs_k
        k1 = 2k - 1
        k2 = 2k
        cone_rows[k1] = cone_rows[k2] = rows
        cone_cols[k1] = (n + p) .+ idxs_k
        cone_cols[k2] = rows
        if Cones.use_dual(cone_k)
            cone_blocks[k1] = cone_k
            cone_blocks[k2] = I
        else
            cone_blocks[k1] = I
            cone_blocks[k2] = cone_k
        end
    end

    system_solver.lhs = BlockMatrix{T}(dim, dim,
        [cone_blocks...,
            model.A', model.G', reshape(model.c, :, 1),
            -model.A, reshape(model.b, :, 1),
            -model.G, reshape(model.h, :, 1), -I,
            -model.c', -model.b', -model.h', -ones(T, 1, 1),
            solver, ones(T, 1, 1)],
        [cone_rows...,
            rc1, rc1, rc1,
            rc2, rc2,
            rc3, rc3, rc3,
            rc4, rc4, rc4, rc4,
            rc6, rc6],
        [cone_cols...,
            rc2, rc3, rc4,
            rc1, rc4,
            rc1, rc4, rc5,
            rc1, rc2, rc3, rc6,
            rc4, rc6],
        )

    return system_solver
end

update_fact(system_solver::NaiveIndirectSystemSolver{T}, solver::Solver{T}) where {T <: Real} = system_solver

function solve_system(system_solver::NaiveIndirectSystemSolver{T}, solver::Solver{T}, sol, rhs) where {T <: Real}
    # TODO tune number of restarts and tolerances etc, ensure initial point in sol helps
    for j in 1:size(rhs, 2)
        rhs_j = view(rhs, :, j)
        sol_j = view(sol, :, j)
        IterativeSolvers.gmres!(sol_j, system_solver.lhs, rhs_j, restart = size(rhs, 1))
    end
    return sol
end

#=
direct sparse
=#

mutable struct NaiveSparseSystemSolver{T <: Real} <: NaiveSystemSolver{T}
    tau_row
    lhs
    hess_idxs
    hess_view_k_j
    sparse_cache::SparseSolverCache # TODO type SparseSolverCache for nonsymmetric systems
    solvecache
    mtt_idx::Int
    function NaiveSparseSystemSolver{T}(; sparse_cache = UMFPACKCache()) where {T <: Real}
        s = new{T}()
        s.sparse_cache = sparse_cache
        return s
    end
end

function load(system_solver::NaiveSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
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

function update_fact(system_solver::NaiveSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
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

function solve_system(system_solver::NaiveSparseSystemSolver{T}, solver::Solver{T}, sol, rhs) where {T <: Real}
    cache = system_solver.sparse_cache
    lhs = system_solver.lhs
    if !cache.analyzed
        analyze_sparse_system(cache, lhs, rhs)
        cache.analyzed = true
    end
    @timeit solver.timer "solve system" solve_sparse_system(cache, sol, lhs, rhs, solver)
    return sol
end

#=
direct dense
=#

mutable struct NaiveDenseSystemSolver{T <: Real} <: NaiveSystemSolver{T}
    tau_row
    lhs
    lhs_copy
    lhs_H_k
    fact_cache
    NaiveDenseSystemSolver{T}() where {T <: Real} = new{T}()
end

function load(system_solver::NaiveDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs
    tau_row = n + p + q + 1
    system_solver.tau_row = tau_row
    dim = n + p + 2q + 2

    system_solver.lhs_copy = T[
        zeros(T,n,n)  model.A'      model.G'              model.c     zeros(T,n,q)           zeros(T,n);
        -model.A      zeros(T,p,p)  zeros(T,p,q)          model.b     zeros(T,p,q)           zeros(T,p);
        -model.G      zeros(T,q,p)  zeros(T,q,q)          model.h     Matrix(-one(T)*I,q,q)  zeros(T,q);
        -model.c'     -model.b'     -model.h'             zero(T)     zeros(T,1,q)           -one(T);
        zeros(T,q,n)  zeros(T,q,p)  Matrix(one(T)*I,q,q)  zeros(T,q)  Matrix(one(T)*I,q,q)   zeros(T,q);
        zeros(T,1,n)  zeros(T,1,p)  zeros(T,1,q)          one(T)      zeros(T,1,q)           one(T);
        ]
    system_solver.lhs = similar(system_solver.lhs_copy)
    # system_solver.fact_cache = HypLUSolveCache(system_solver.sol, system_solver.lhs, rhs)

    function view_H_k(cone_k, idxs_k)
        rows = tau_row .+ idxs_k
        cols = Cones.use_dual(cone_k) ? (n + p) .+ idxs_k : rows
        return view(system_solver.lhs, rows, cols)
    end
    system_solver.lhs_H_k = [view_H_k(cone_k, idxs_k) for (cone_k, idxs_k) in zip(cones, cone_idxs)]

    return system_solver
end

function update_fact(system_solver::NaiveDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    copyto!(system_solver.lhs, system_solver.lhs_copy)
    system_solver.lhs[end, system_solver.tau_row] = solver.mu / solver.tau / solver.tau
    for (k, cone_k) in enumerate(solver.model.cones)
        copyto!(system_solver.lhs_H_k[k], Cones.hess(cone_k))
    end
    system_solver.fact_cache = lu!(system_solver.lhs) # TODO use wrapped lapack function
    return system_solver
end

function solve_system(system_solver::NaiveDenseSystemSolver{T}, solver::Solver{T}, sol, rhs) where {T <: Real}
    # if !hyp_lu_solve!(system_solver.fact_cache, sol, lhs, rhs)
    #     @warn("numerical failure: could not fix linear solve failure")
    # end
    ldiv!(sol, system_solver.fact_cache, rhs) # TODO use wrapped lapack function
    return sol
end
