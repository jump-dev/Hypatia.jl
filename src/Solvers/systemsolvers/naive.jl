#=
naive linear system solver
solves 6x6 system without reductions
=#

abstract type NaiveSystemSolver{T <: Real} <: SystemSolver{T} end

#=
direct sparse
=#

mutable struct NaiveSparseSystemSolver{T <: Real} <: NaiveSystemSolver{T}
    lhs::SparseMatrixCSC
    fact_cache::SparseNonSymCache{T}
    hess_idxs::Vector
    mtt_idx::Int
    function NaiveSparseSystemSolver{T}(; fact_cache::SparseNonSymCache{T} = SparseNonSymCache{T}()) where {T <: Real}
        s = new{T}()
        s.fact_cache = fact_cache
        return s
    end
end

function load(system_solver::NaiveSparseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    system_solver.fact_cache.analyzed = false
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = n + p + q + 1
    cones = model.cones
    cone_idxs = model.cone_idxs

    # form sparse LHS without Hessians in s row
    lhs = T[
        spzeros(T, n, n)  model.A'          model.G'                  model.c        spzeros(T, n, q)           spzeros(T, n);
        -model.A          spzeros(T, p, p)  spzeros(T, p, q)          model.b        spzeros(T, p, q)           spzeros(T, p);
        -model.G          spzeros(T, q, p)  spzeros(T, q, q)          model.h        sparse(-one(T) * I, q, q)  spzeros(T, q);
        -model.c'         -model.b'         -model.h'                 zero(T)        spzeros(T, 1, q)           -one(T);
        spzeros(T, q, n)  spzeros(T, q, p)  sparse(one(T) * I, q, q)  spzeros(T, q)  sparse(one(T) * I, q, q)   spzeros(T, q);
        spzeros(T, 1, n)  spzeros(T, 1, p)  spzeros(T, 1, q)          one(T)         spzeros(T, 1, q)           one(T);
        ]
    dropzeros!(lhs)
    (Is, Js, Vs) = findnz(lhs)

    # add I, J, V for Hessians
    if isempty(cones)
        hess_nz_total = 0
    else
        hess_nz_total = sum(Cones.hess_nz_count(cone_k) for cone_k in cones)
    end
    H_Is = Vector{Int}(undef, hess_nz_total)
    H_Js = Vector{Int}(undef, hess_nz_total)
    offset = 1
    for (cone_k, idxs_k) in zip(cones, cone_idxs)
        z_start_k = n + p + first(idxs_k) - 1
        s_start_k = tau_row + first(idxs_k) - 1
        H_start_k = Cones.use_dual_barrier(cone_k) ? z_start_k : s_start_k
        for j in 1:Cones.dimension(cone_k)
            nz_rows_kj = s_start_k .+ Cones.hess_nz_idxs_col(cone_k, j)
            len_kj = length(nz_rows_kj)
            IJV_idxs = offset:(offset + len_kj - 1)
            offset += len_kj
            @. @views H_Is[IJV_idxs] = nz_rows_kj
            @. @views H_Js[IJV_idxs] = H_start_k + j
        end
    end
    append!(Is, H_Is)
    append!(Js, H_Js)
    append!(Vs, ones(T, hess_nz_total))

    # prefer conversions of integer types to happen here than inside external wrappers
    dim = size(lhs, 1)
    # integer type supported by the sparse system solver library to be used
    Ti = int_type(system_solver.fact_cache)
    Is = convert(Vector{Ti}, Is)
    Js = convert(Vector{Ti}, Js)
    lhs = system_solver.lhs = sparse(Is, Js, Vs, dim, dim)

    # cache indices of nonzeros of Hessians in sparse LHS nonzeros vector
    system_solver.hess_idxs = [Vector{Union{UnitRange, Vector{Int}}}(undef, Cones.dimension(cone_k)) for cone_k in cones]
    for (k, cone_k) in enumerate(cones)
        idxs_k = cone_idxs[k]
        z_start_k = n + p + first(idxs_k) - 1
        s_start_k = tau_row + first(idxs_k) - 1
        H_start_k = Cones.use_dual_barrier(cone_k) ? z_start_k : s_start_k
        for j in 1:Cones.dimension(cone_k)
            col = H_start_k + j
            # get nonzero rows in the current column of the LHS
            col_idx_start = lhs.colptr[col]
            nz_rows = lhs.rowval[col_idx_start:(lhs.colptr[col + 1] - 1)]
            # get nonzero rows in column j of the Hessian
            nz_hess_indices = Cones.hess_nz_idxs_col(cone_k, j)
            # get index corresponding to first nonzero Hessian element of the current column of the LHS
            first_H = findfirst(isequal(s_start_k + first(nz_hess_indices)), nz_rows)
            # indices of nonzero values for cone k column j
            system_solver.hess_idxs[k][j] = (col_idx_start + first_H - 2) .+ (1:length(nz_hess_indices))
        end
    end

    # get mu/tau/tau index
    system_solver.mtt_idx = lhs.colptr[tau_row + 1] - 1

    return system_solver
end

function update_lhs(system_solver::NaiveSparseSystemSolver, solver::Solver)
    for (k, cone_k) in enumerate(solver.model.cones)
        H_k = Cones.hess(cone_k)
        for j in 1:Cones.dimension(cone_k)
            nz_rows = Cones.hess_nz_idxs_col(cone_k, j)
            @views copyto!(system_solver.lhs.nzval[system_solver.hess_idxs[k][j]], H_k[nz_rows, j])
        end
    end
    tau = solver.point.tau[]
    system_solver.lhs.nzval[system_solver.mtt_idx] = solver.mu / tau / tau # NOTE: mismatch when using NT for kaptau

    update_fact(system_solver.fact_cache, system_solver.lhs)

    return system_solver
end

function solve_system(
    system_solver::NaiveSparseSystemSolver,
    solver::Solver{T},
    sol::Point{T},
    rhs::Point{T},
    ::T,
    ) where {T <: Real}
    inv_prod(system_solver.fact_cache, sol.vec, system_solver.lhs, rhs.vec)
    return sol
end

#=
direct dense
=#

mutable struct NaiveDenseSystemSolver{T <: Real} <: NaiveSystemSolver{T}
    tau_row::Int
    lhs::Matrix{T}
    fact_cache::DenseNonSymCache{T}
    lhs_H_k::Vector
    function NaiveDenseSystemSolver{T}(; fact_cache::DenseNonSymCache{T} = DenseNonSymCache{T}()) where {T <: Real}
        system_solver = new{T}()
        system_solver.fact_cache = fact_cache
        return system_solver
    end
end

function load(system_solver::NaiveDenseSystemSolver{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs
    system_solver.tau_row = n + p + q + 1

    system_solver.lhs = T[
        zeros(T, n, n)  model.A'        model.G'                  model.c      zeros(T, n, q)             zeros(T, n);
        -model.A        zeros(T, p, p)  zeros(T, p, q)            model.b      zeros(T, p, q)             zeros(T, p);
        -model.G        zeros(T, q, p)  zeros(T, q, q)            model.h      Matrix(-one(T) * I, q, q)  zeros(T, q);
        -model.c'       -model.b'       -model.h'                 zero(T)      zeros(T, 1, q)             -one(T);
        zeros(T, q, n)  zeros(T, q, p)  Matrix(one(T) * I, q, q)  zeros(T, q)  Matrix(one(T) * I, q, q)   zeros(T, q);
        zeros(T, 1, n)  zeros(T, 1, p)  zeros(T, 1, q)            one(T)       zeros(T, 1, q)             one(T);
        ]

    function view_H_k(cone_k, idxs_k)
        rows = system_solver.tau_row .+ idxs_k
        cols = Cones.use_dual_barrier(cone_k) ? (n + p) .+ idxs_k : rows
        return view(system_solver.lhs, rows, cols)
    end
    system_solver.lhs_H_k = [view_H_k(cone_k, idxs_k) for (cone_k, idxs_k) in zip(cones, cone_idxs)]

    load_matrix(system_solver.fact_cache, system_solver.lhs)

    return system_solver
end

function update_lhs(system_solver::NaiveDenseSystemSolver, solver::Solver)
    for (cone_k, lhs_H_k) in zip(solver.model.cones, system_solver.lhs_H_k)
        copyto!(lhs_H_k, Cones.hess(cone_k))
    end
    tau = solver.point.tau[]
    system_solver.lhs[end, system_solver.tau_row] = solver.mu / tau / tau # NOTE: mismatch when using NT for kaptau

    update_fact(system_solver.fact_cache, system_solver.lhs)

    return system_solver
end

function solve_system(
    system_solver::NaiveDenseSystemSolver,
    solver::Solver,
    sol::Point{T},
    rhs::Point{T},
    ::T,
    ) where {T <: Real}
    copyto!(sol.vec, rhs.vec)
    inv_prod(system_solver.fact_cache, sol.vec)
    return sol
end
