#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

TODO play around with SuiteSparse parameters e.g. iterative refinement
TODO try not to fallback to lu
=#

max_num_threads = length(Sys.cpu_info())
ENV["OMP_NUM_THREADS"] = max_num_threads
import Pardiso
import SuiteSparse.UMFPACK
import SuiteSparse.CHOLMOD
import Pardiso: PardisoSolver, pardiso
import SuiteSparse.UMFPACK: UmfpackLU
import SuiteSparse.CHOLMOD: Factor

abstract type SparseSystemSolver <: SystemSolver{Float64} end

abstract type SparseSolverCache end

mutable struct PardisoCache <: SparseSolverCache
    use_symmetric::Bool
    analyzed::Bool
    ps::PardisoSolver
end
PardisoCache(use_symmetric::Bool) = PardisoCache(use_symmetric, false, PardisoSolver())

mutable struct UMFPACKCache <: SparseSolverCache
    analyzed::Bool
    fact::UmfpackLU
    function UMFPACKCache()
        cache = new()
        cache.analyzed = false
        return cache
    end
end

mutable struct CHOLMODCache <: SparseSolverCache
    analyzed::Bool
    fact::Factor
    backup_fact::UmfpackLU
    function CHOLMODCache()
        cache = new()
        cache.analyzed = false
        return cache
    end
end

reset_sparse_cache(cache::SparseSolverCache) = (cache.analyzed = false; cache)

function analyze_sparse_system(cache::PardisoCache, A::SparseMatrixCSC, b::Matrix)
    ps = cache.ps
    if cache.use_symmetric
        Pardiso.set_matrixtype!(ps, Pardiso.REAL_SYM_INDEF)
    end
    Pardiso.pardisoinit(ps)
    Pardiso.set_iparm!(ps, 1, 1)
    # if !cache.use_symmetric
        Pardiso.set_iparm!(ps, 12, 1)
    # end
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
    # TODO investigate why we need this
    LinearAlgebra.copytri!(A, 'L')
    pardiso(ps, A, b)
    return
end

function analyze_sparse_system(cache::UMFPACKCache, A::SparseMatrixCSC, ::Matrix)
    cache.fact = lu(A)
    return
end

function analyze_sparse_system(cache::CHOLMODCache, A::SparseMatrixCSC, ::Matrix)
    # A.rowval = convert(Vecdtor{Float})
    cache.fact = CHOLMOD.ldlt(Symmetric(A, :L), check = false)
    return
end

function solve_sparse_system(cache::PardisoCache, x::Matrix, A::SparseMatrixCSC, b::Matrix, solver)
    ps = cache.ps
    Pardiso.set_phase!(ps, Pardiso.NUM_FACT_SOLVE_REFINE)
    @timeit solver.timer "solve" if cache.use_symmetric
        pardiso(ps, x, A, b)
    else
        pardiso(ps, x, A, b)
    end
    return x
end

function solve_sparse_system(cache::UMFPACKCache, x::Matrix, A::SparseMatrixCSC, b::Matrix, solver)
    fact = cache.fact
    # TODO this is a hack around lack of interface https://github.com/JuliaLang/julia/issues/33323
    copyto!(fact.nzval, A.nzval)
    fact.numeric = C_NULL
    @timeit solver.timer "solve" ldiv!(x, fact, b)
    return x
end

function solve_sparse_system(cache::CHOLMODCache, x::Matrix, A::SparseMatrixCSC, b::Matrix, solver)
    A_symm = Symmetric(A, :L)
    ldlt!(cache.fact, A_symm, check = false)
    if issuccess(cache.fact)
        @timeit solver.timer "solve" x .= cache.fact \ b
    else
        cache.backup_fact = lu(SparseMatrixCSC{eltype(A), SuiteSparse.CHOLMOD.SuiteSparse_long}(A_symm))
        x .= cache.backup_fact \ b
    end
    return x
end

function release_sparse_cache(cache::PardisoCache)
    ps = cache.ps
    Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)
    return
end

release_sparse_cache(::UMFPACKCache) = nothing

release_sparse_cache(::CHOLMODCache) = nothing

release_sparse_cache(s::SparseSystemSolver) = release_sparse_cache(s.sparse_cache)

release_sparse_cache(s::SystemSolver) = nothing

function add_I_J_V(
    offset::Int,
    Is::Vector{<: Integer},
    Js::Vector{<: Integer},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    vec::Vector{Float64},
    trans::Bool,
    )
    n = length(vec)
    if !isempty(vec)
        if trans
            @views Is[offset:(offset + n - 1)] .= start_row + 1
            @views Js[offset:(offset + n - 1)] .= (start_col + 1):(start_col + n)
        else
            @views Is[offset:(offset + n - 1)] .= (start_row + 1):(start_row + n)
            @views Js[offset:(offset + n - 1)] .= start_col + 1
        end
        Vs[offset:(offset + n - 1)] .= vec
    end
    return offset + n
end

function add_I_J_V(
    offset::Int,
    Is::Vector{<: Integer},
    Js::Vector{<: Integer},
    Vs::Vector{Float64},
    start_rows::Vector{Int},
    start_cols::Vector{Int},
    vecs::Vector{Vector{Float64}},
    trans::Vector{Bool},
    )
    for (r, c, v, t) in zip(start_rows, start_cols, vecs, trans)
        offset = add_I_J_V(offset, Is, Js, Vs, r, c, v, t)
    end
    return offset
end

function add_I_J_V(
    offset::Int,
    Is::Vector{<: Integer},
    Js::Vector{<: Integer},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    mat::SparseMatrixCSC,
    trans::Bool,
    )
    for j in 1:mat.n
        col_idxs = mat.colptr[j]:(mat.colptr[j + 1] - 1)
        rows = view(mat.rowval, col_idxs)
        vals = view(mat.nzval, col_idxs)
        m = length(rows)
        if trans
            @views Is[offset:(offset + m - 1)] .= start_row + j
            @views Js[offset:(offset + m - 1)] .= start_col .+ rows
        else
            @views Is[offset:(offset + m - 1)] .= start_row .+ rows
            @views Js[offset:(offset + m - 1)] .= start_col + j
        end
        @views Vs[offset:(offset + m - 1)] .= vals
        offset += m
    end
    return offset
end

function add_I_J_V(
    offset::Int,
    Is::Vector{<: Integer},
    Js::Vector{<: Integer},
    Vs::Vector{Float64},
    start_rows::Vector{Int},
    start_cols::Vector{Int},
    mats::Vector{<: SparseMatrixCSC},
    trans::Vector{Bool},
    )
    for (r, c, m, t) in zip(start_rows, start_cols, mats, trans)
        offset = add_I_J_V(offset, Is, Js, Vs, r, c, m, t)
    end
    return offset
end

function add_I_J_V(
    offset::Int,
    Is::Vector{<: Integer},
    Js::Vector{<: Integer},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    cone::Cones.Cone,
    use_inv::Bool)
    for j in 1:Cones.dimension(cone)
        nz_rows = (use_inv ? Cones.inv_hess_nz_idxs_j(cone, j) : Cones.hess_nz_idxs_j(cone, j))
        n = length(nz_rows)
        @. Is[offset:(offset + n - 1)] = start_row + nz_rows
        @. Js[offset:(offset + n - 1)] = j + start_col
        @. Vs[offset:(offset + n - 1)] = 1
        offset += n
    end
    return offset
end
