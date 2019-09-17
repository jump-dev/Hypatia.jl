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

TODO treat epinorminf hessians as sparse
=#

import Pardiso

struct DefaultSparseSolver end
SparseSystemSolver = Union{DefaultSparseSolver, Pardiso.PardisoSolver}

function reset_sparse_solver(ps::Pardiso.PardisoSolver)
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS_NUM_FACT_SOLVE_REFINE)
    return
end
reset_sparse_solver(::DefaultSparseSolver) = nothing

function analyze_sparse_system(ps::Pardiso.PardisoSolver, lhs::SparseMatrixCSC, rhs::Matrix)
    Pardiso.pardisoinit(ps)
    Pardiso.set_iparm!(ps, 1, 1)
    Pardiso.set_iparm!(ps, 12, 1)
    Pardiso.set_iparm!(ps, 6, 1)
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
    Pardiso.pardiso(ps, lhs, rhs)
    return
end
analyze_sparse_system(::DefaultSparseSolver, ::SparseMatrixCSC, ::Matrix) = nothing

function solve_sparse_system(ps::Pardiso.PardisoSolver, sol::Matrix, lhs::SparseMatrixCSC, rhs::Matrix, solver)
    if Pardiso.get_phase(ps) == Pardiso.ANALYSIS_NUM_FACT_SOLVE_REFINE
        @timeit solver.timer "analyze" analyze_sparse_system(ps, lhs, rhs)
    end
    Pardiso.set_phase!(ps, Pardiso.NUM_FACT_SOLVE_REFINE)
    @timeit solver.timer "solve" Pardiso.pardiso(ps, sol, lhs, rhs)
    return sol
end
solve_sparse_system(ps::DefaultSparseSolver, ::Matrix, lhs::SparseMatrixCSC, rhs::Matrix, ::Any) = (rhs .= lu(lhs) \ rhs)

function free_sparse_solver_memory(ps::Pardiso.PardisoSolver)
    Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(ps, sol, lhs, rhs)
    return
end
free_sparse_solver_memory(::DefaultSparseSolver) = nothing

mutable struct NaiveSparseSystemSolver{T <: Real} <: SystemSolver{T}
    use_iterative::Bool
    use_sparse::Bool

    solver::Solver{T}

    x1
    x2
    y1
    y2
    z1
    z2
    tau_row::Int
    mtt_idx::Int
    s1
    s2
    s1_k
    s2_k

    lhs
    hess_idxs
    hess_view_k_j
    sparse_solver::SparseSystemSolver

    sol::Matrix{T}

    rhs::Matrix{T}
    prevsol1::Vector{T}
    prevsol2::Vector{T}

    solvesol
    solvecache

    function NaiveSparseSystemSolver{T}(;
            use_iterative::Bool = false,
            use_sparse::Bool = false,
            sparse_solver = Pardiso.PardisoSolver() # DefaultSparseSolver()
            ) where {T <: Real}
        system_solver = new{T}()
        system_solver.use_iterative = use_iterative
        system_solver.use_sparse = use_sparse
        system_solver.sparse_solver = sparse_solver
        return system_solver
    end
end

free_sparse_solver_memory(s::NaiveSparseSystemSolver) = free_sparse_solver_memory(s.sparse_solver)

function load(system_solver::NaiveSparseSystemSolver{Float64}, solver::Solver{Float64})
    T = Float64
    @timeit solver.timer "load" begin
    system_solver.solver = solver

    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    dim = n + p + 2q + 2

    reset_sparse_solver(system_solver.sparse_solver)

    rhs = zeros(T, dim, 2)
    system_solver.rhs = rhs
    system_solver.sol = similar(rhs)
    rows = 1:n
    system_solver.x1 = view(rhs, rows, 1)
    system_solver.x2 = view(rhs, rows, 2)
    rows = (n + 1):(n + p)
    system_solver.y1 = view(rhs, rows, 1)
    system_solver.y2 = view(rhs, rows, 2)
    rows = (n + p + 1):(n + p + q)
    system_solver.z1 = view(rhs, rows, 1)
    system_solver.z2 = view(rhs, rows, 2)
    tau_row = n + p + q + 1
    system_solver.tau_row = tau_row
    rows = tau_row .+ (1:q)
    system_solver.s1 = view(rhs, rows, 1)
    system_solver.s2 = view(rhs, rows, 2)
    system_solver.s1_k = [view(rhs, tau_row .+ model.cone_idxs[k], 1) for k in eachindex(model.cones)]
    system_solver.s2_k = [view(rhs, tau_row .+ model.cone_idxs[k], 2) for k in eachindex(model.cones)]

    # TODO remove
    model.A = sparse(model.A)
    model.G = sparse(model.G)

    dropzeros!(model.A)
    dropzeros!(model.G)

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

    hess_nnzs = sum(Cones.dimension(cone_k) + Cones.hess_nnzs(cone_k) for cone_k in model.cones)
    total_nnz = 2 * (nnz(sparse(model.A)) + nnz(sparse(model.G)) + n + p + q + 1) + q + 1 + hess_nnzs
    Is = Vector{Int32}(undef, total_nnz)
    Js = Vector{Int32}(undef, total_nnz)
    Vs = Vector{T}(undef, total_nnz)

    function add_I_J_V(k, start_row, start_col, vec::Vector{Float64}, trans::Bool = false)
        n = length(vec)
        if !isempty(vec)
            if trans
                Is[k:(k + n - 1)] .= start_row + 1
                Js[k:(k + n - 1)] .= (start_col + 1):(start_col + n)
            else
                Is[k:(k + n - 1)] .= (start_row + 1):(start_row + n)
                Js[k:(k + n - 1)] .= start_col + 1
            end
            Vs[k:(k + n - 1)] .= vec
        end
        return k + n
    end

    function add_I_J_V(k, start_row, start_col, mat)
        if !isempty(mat)
            for (i, j, v) in zip(findnz(mat)...)
                Is[k] = i + start_row
                Js[k] = j + start_col
                Vs[k] = v
                k += 1
            end
        end
        return k
    end

    rc1 = 0
    rc2 = n
    rc3 = n + p
    rc4 = n + p + q
    rc5 = n + p + q + 1
    rc6 = dim - 1
    # count of nonzeros added so far
    offset = 1
    @timeit solver.timer "setup lhs" begin
    # set up all nonzero elements apart from Hessians
    offset = add_I_J_V(offset, rc1, rc2, sparse(model.A')) # slow but doesn't allocate much
    offset = add_I_J_V(offset, rc1, rc3, sparse(model.G'))
    offset = add_I_J_V(offset, rc1, rc4, model.c)
    offset = add_I_J_V(offset, rc2, rc1, -model.A)
    offset = add_I_J_V(offset, rc2, rc4, model.b)
    offset = add_I_J_V(offset, rc3, rc1, -model.G)
    offset = add_I_J_V(offset, rc3, rc4, model.h)
    offset = add_I_J_V(offset, rc3, rc5, sparse(-one(T) * I, q, q))
    offset = add_I_J_V(offset, rc4, rc1, -model.c, true)
    offset = add_I_J_V(offset, rc4, rc2, -model.b, true)
    offset = add_I_J_V(offset, rc4, rc3, -model.h, true)
    offset = add_I_J_V(offset, rc4, rc6, -[one(T)])
    offset = add_I_J_V(offset, rc6, rc4, [one(T)])
    offset = add_I_J_V(offset, rc6, rc6, [one(T)])

    # add I, J, V for Hessians
    @timeit solver.timer "setup hess lhs" begin
    nz_rows_added = 0
    for (k, cone_k) in enumerate(model.cones)
        cone_dim = Cones.dimension(cone_k)
        rows = rc5 + nz_rows_added
        dual_cols = rc3 + nz_rows_added
        is_dual = Cones.use_dual(cone_k)
        # add dense matrix in one placeholder block, identity in the other
        H_cols = (is_dual ? dual_cols : rows)
        id_cols = (is_dual ? rows : dual_cols)
        offset = add_I_J_V(offset, rows, H_cols, Cones.hess_sparsity_pattern(cone_k))
        offset = add_I_J_V(offset, rows, id_cols, sparse(one(T) * I, cone_dim, cone_dim))
        nz_rows_added += cone_dim
    end
    end # hess timing
    end # setup lhs timing
    @assert offset == total_nnz + 1

    @timeit solver.timer "build sparse" system_solver.lhs = sparse(Is, Js, Vs, Int32(dim), Int32(dim))

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
            col_idx_start = system_solver.lhs.colptr[col]
            col_idx_end = system_solver.lhs.colptr[col + 1] - 1
            nz_rows = system_solver.lhs.rowval[col_idx_start:col_idx_end]
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
    system_solver.mtt_idx = system_solver.lhs.colptr[rc4 + 2] - 1

    system_solver.hess_view_k_j = [[view(cone_k.hess, :, j) for j in 1:Cones.dimension(cone_k)] for cone_k in model.cones]

    end # load timing

    return system_solver
end

function get_combined_directions(system_solver::NaiveSparseSystemSolver{T}) where {T <: Real}
    solver = system_solver.solver
    model = solver.model
    cones = model.cones
    lhs = system_solver.lhs
    sol = system_solver.sol
    sparse_solver = system_solver.sparse_solver

    rhs = system_solver.rhs
    tau_row = system_solver.tau_row
    x1 = system_solver.x1
    x2 = system_solver.x2
    y1 = system_solver.y1
    y2 = system_solver.y2
    z1 = system_solver.z1
    z2 = system_solver.z2
    s1 = system_solver.s1
    s2 = system_solver.s2
    s1_k = system_solver.s1_k
    s2_k = system_solver.s2_k

    sqrtmu = sqrt(solver.mu)
    mtt = solver.mu / solver.tau / solver.tau

    # update rhs matrix
    x1 .= solver.x_residual
    x2 .= zero(T)
    y1 .= solver.y_residual
    y2 .= zero(T)
    z1 .= solver.z_residual
    z2 .= zero(T)
    rhs[tau_row, 1] = solver.kap + solver.primal_obj_t - solver.dual_obj_t
    rhs[tau_row, 2] = zero(T)
    for k in eachindex(cones)
        duals_k = solver.point.dual_views[k]
        grad_k = Cones.grad(cones[k])
        @. s1_k[k] = -duals_k
        @. s2_k[k] = -duals_k - grad_k * sqrtmu
    end
    rhs[end, 1] = -solver.kap
    rhs[end, 2] = -solver.kap + solver.mu / solver.tau

    # @timeit solver.timer "modify views" begin
    for (k, cone_k) in enumerate(cones)
        Cones.update_hess(cone_k)
        for j in 1:Cones.dimension(cone_k)
            if isa(cone_k, Cones.OrthantCone)
                @views copyto!(system_solver.lhs.nzval[system_solver.hess_idxs[k][j]], Cones.hess(cone_k)[j, j])
            else
                # @views copyto!(system_solver.lhs.nzval[system_solver.hess_idxs[k][j]], model.cones[k].hess[:, j])
                @views copyto!(system_solver.lhs.nzval[system_solver.hess_idxs[k][j]], system_solver.hess_view_k_j[k][j])
            end
        end
    end
    # end # time views
    system_solver.lhs.nzval[system_solver.mtt_idx] = mtt
    @timeit solver.timer "solve system" solve_sparse_system(sparse_solver, sol, lhs, rhs, solver)

    return (x1, x2, y1, y2, z1, z2, rhs[tau_row, 1], rhs[tau_row, 2], s1, s2, rhs[end, 1], rhs[end, 2])
end
