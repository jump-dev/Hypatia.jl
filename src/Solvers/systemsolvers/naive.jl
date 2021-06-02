#=
naive linear system solver
solves 6x6 system without reductions
=#

abstract type NaiveSystemSolver{T <: Real} <: SystemSolver{T} end

#=
direct sparse
=#

mutable struct NaiveSparseSystemSolver{T <: Real} <: NaiveSystemSolver{T}
    lhs::SparseMatrixCSC{T}
    fact_cache::SparseNonSymCache{T}
    hess_idxs::Vector
    mtt_idx::Int
    function NaiveSparseSystemSolver{T}(; fact_cache::SparseNonSymCache{T} =
        SparseNonSymCache{T}()) where {T <: Real}
        s = new{T}()
        s.fact_cache = fact_cache
        return s
    end
end

function load(
    syssolver::NaiveSparseSystemSolver{T},
    solver::Solver{T},
    ) where {T <: Real}
    syssolver.fact_cache.analyzed = false
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    tau_row = n + p + q + 1
    cones = model.cones
    cone_idxs = model.cone_idxs

    # form sparse LHS without Hessians in s row
    # TODO check for inefficiency, may need to implement more manually
    spz(a, b) = spzeros(T, a, b)
    spi(a, b) = sparse(one(T) * I, a, b)
    lhs = hvcat((5, 4, 4, 5, 3, 4),
        spz(n, n), model.A', model.G', model.c, spz(n, q + 1),
        -model.A, spz(p, p + q), model.b, spz(p, q + 1),
        -model.G, spz(q, p + q), model.h, -spi(q, q + 1),
        -model.c', -model.b', -model.h', spz(1, 1 + q), -1,
        spz(q, n + p), spi(q, q + 1), spi(q, q + 1),
        spz(1, n + p + q), 1, spz(1, q), 1)
    @assert lhs isa SparseMatrixCSC{T}
    dropzeros!(lhs)
    (Is, Js, Vs) = findnz(lhs)

    # add I, J, V for Hessians
    hess_nz_total = (isempty(cones) ? 0 : sum(
        Cones.hess_nz_count(cone_k) for cone_k in cones))
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
            @. H_Is[IJV_idxs] = nz_rows_kj
            @. H_Js[IJV_idxs] = H_start_k + j
        end
    end
    append!(Is, H_Is)
    append!(Js, H_Js)
    append!(Vs, ones(T, hess_nz_total))

    # convert integer types here rather than inside external wrappers
    dim = size(lhs, 1)
    # integer type supported by the sparse system solver library to be used
    Ti = int_type(syssolver.fact_cache)
    Is = convert(Vector{Ti}, Is)
    Js = convert(Vector{Ti}, Js)
    lhs = syssolver.lhs = sparse(Is, Js, Vs, dim, dim)

    # cache indices of nonzeros of Hessians in sparse LHS nonzeros vector
    syssolver.hess_idxs = [Vector{Union{UnitRange, Vector{Int}}}(
        undef, Cones.dimension(cone_k)) for cone_k in cones]
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
            # get index corresponding to first nonzero element of the current col
            first_H = findfirst(isequal(
                s_start_k + first(nz_hess_indices)), nz_rows)
            # indices of nonzero values for cone k column j
            syssolver.hess_idxs[k][j] = (col_idx_start + first_H - 2) .+
                (1:length(nz_hess_indices))
        end
    end

    # get mu/tau/tau index
    syssolver.mtt_idx = lhs.colptr[tau_row + 1] - 1

    return syssolver
end

function update_lhs(syssolver::NaiveSparseSystemSolver, solver::Solver)
    for (k, cone_k) in enumerate(solver.model.cones)
        H_k = Cones.hess(cone_k)
        for j in 1:Cones.dimension(cone_k)
            nz_rows = Cones.hess_nz_idxs_col(cone_k, j)
            @views copyto!(syssolver.lhs.nzval[
                syssolver.hess_idxs[k][j]], H_k[nz_rows, j])
        end
    end
    tau = solver.point.tau[]
    syssolver.lhs.nzval[syssolver.mtt_idx] = solver.mu / tau / tau

    solver.time_upfact += @elapsed update_fact(syssolver.fact_cache,
        syssolver.lhs)

    return syssolver
end

function solve_system(
    syssolver::NaiveSparseSystemSolver,
    solver::Solver{T},
    sol::Point{T},
    rhs::Point{T},
    ) where {T <: Real}
    inv_prod(syssolver.fact_cache, sol.vec, syssolver.lhs, rhs.vec)
    return sol
end

#=
direct dense
=#

mutable struct NaiveDenseSystemSolver{T <: Real} <: NaiveSystemSolver{T}
    tau_row::Int
    lhs::Matrix{T}
    lhs_fact::Matrix{T}
    fact::LU{T, Matrix{T}}
    lhs_H_k::Vector
    function NaiveDenseSystemSolver{T}() where {T <: Real}
        syssolver = new{T}()
        return syssolver
    end
end

function load(
    syssolver::NaiveDenseSystemSolver{T},
    solver::Solver{T},
    ) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs
    syssolver.tau_row = n + p + q + 1

    # TODO check for inefficiency, may need to implement more manually
    dz(a, b) = zeros(T, a, b)
    di(a, b) = Matrix(one(T) * I, a, b)
    lhs = hvcat((5, 4, 4, 5, 3, 4),
        dz(n, n), model.A', model.G', model.c, dz(n, q + 1),
        -model.A, dz(p, p + q), model.b, dz(p, q + 1),
        -model.G, dz(q, p + q), model.h, -di(q, q + 1),
        -model.c', -model.b', -model.h', dz(1, 1 + q), -1,
        dz(q, n + p), di(q, q + 1), di(q, q + 1),
        dz(1, n + p + q), 1, dz(1, q), 1)
    @assert lhs isa Matrix{T}
    syssolver.lhs = lhs
    syssolver.lhs_fact = zero(lhs)

    function view_H_k(cone_k, idxs_k)
        rows = syssolver.tau_row .+ idxs_k
        cols = Cones.use_dual_barrier(cone_k) ? (n + p) .+ idxs_k : rows
        return view(syssolver.lhs, rows, cols)
    end
    syssolver.lhs_H_k = [view_H_k(cone_k, idxs_k) for
        (cone_k, idxs_k) in zip(cones, cone_idxs)]

    return syssolver
end

function update_lhs(syssolver::NaiveDenseSystemSolver, solver::Solver)
    for (cone_k, lhs_H_k) in zip(solver.model.cones, syssolver.lhs_H_k)
        copyto!(lhs_H_k, Cones.hess(cone_k))
    end
    tau = solver.point.tau[]
    syssolver.lhs[end, syssolver.tau_row] = solver.mu / tau / tau

    solver.time_upfact += @elapsed syssolver.fact =
        nonsymm_fact_copy!(syssolver.lhs_fact, syssolver.lhs)

    if !issuccess(syssolver.fact)
        println("nonsymmetric linear system factorization failed")
    end

    return syssolver
end

function solve_system(
    syssolver::NaiveDenseSystemSolver,
    solver::Solver,
    sol::Point{T},
    rhs::Point{T},
    ) where {T <: Real}
    ldiv!(sol.vec, syssolver.fact, rhs.vec)
    return sol
end
