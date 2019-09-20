#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

TODO play around with SuiteSparse parameters e.g. iterative refinement
=#

max_num_threads = length(Sys.cpu_info())
ENV["OMP_NUM_THREADS"] = max_num_threads
import Pardiso
import SuiteSparse.UMFPACK
import Pardiso: PardisoSolver, pardiso
import SuiteSparse.UMFPACK: UmfpackLU
import SuiteSparse.CHOLMOD: Factor

abstract type SparseSolverCache end

mutable struct PardisoCache <: SparseSolverCache
    analyzed::Bool
    ps::PardisoSolver
end
PardisoCache() = PardisoCache(false, PardisoSolver())

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
    function CHOLMODCache()
        cache = new()
        cache.analyzed = false
        return cache
    end
end

reset_sparse_cache(cache::SparseSolverCache) = (cache.analyzed = false; cache)

function analyze_sparse_system(cache::PardisoCache, A::SparseMatrixCSC, b::Matrix)
    ps = cache.ps
    Pardiso.pardisoinit(ps)
    Pardiso.set_iparm!(ps, 1, 1)
    Pardiso.set_iparm!(ps, 12, 1)
    Pardiso.set_phase!(ps, Pardiso.ANALYSIS)
    pardiso(ps, A, b)
    return
end

function analyze_sparse_system(cache::UMFPACKCache, A::SparseMatrixCSC, ::Matrix)
    cache.fact = lu(A)
    return
end

function solve_sparse_system(cache::PardisoCache, x::Matrix, A::SparseMatrixCSC, b::Matrix, solver)
    ps = cache.ps
    Pardiso.set_phase!(ps, Pardiso.NUM_FACT_SOLVE_REFINE)
    @timeit solver.timer "solve" pardiso(ps, x, A, b)
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

function release_sparse_cache(cache::PardisoCache)
    ps = cache.ps
    Pardiso.set_phase!(ps, Pardiso.RELEASE_ALL)
    pardiso(ps)
    return
end

release_sparse_cache(::UMFPACKCache) = nothing

release_sparse_cache(s::SystemSolver) = release_sparse_cache(s.sparse_cache)

function add_I_J_V(
    offset::Int,
    Is::Vector{Int32},
    Js::Vector{Int32},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    vec::Vector{Float64},
    trans::Bool,
    )
    n = length(vec)
    if !isempty(vec)
        if trans
            Is[offset:(offset + n - 1)] .= start_row + 1
            Js[offset:(offset + n - 1)] .= (start_col + 1):(start_col + n)
        else
            Is[offset:(offset + n - 1)] .= (start_row + 1):(start_row + n)
            Js[offset:(offset + n - 1)] .= start_col + 1
        end
        Vs[offset:(offset + n - 1)] .= vec
    end
    return offset + n
end

function add_I_J_V(
    offset::Int,
    Is::Vector{Int32},
    Js::Vector{Int32},
    Vs::Vector{Float64},
    start_rows::Vector{Int},
    start_cols::Vector{Int},
    vecs::Vector{Vector{Float64}},
    trans::Vector{Bool}
    )
    for (r, c, v, t) in zip(start_rows, start_cols, vecs, trans)
        offset = add_I_J_V(offset, Is, Js, Vs, r, c, v, t)
    end
    return offset
end

function add_I_J_V(
    offset::Int,
    Is::Vector{Int32},
    Js::Vector{Int32},
    Vs::Vector{Float64},
    start_row::Int,
    start_col::Int,
    mat::SparseMatrixCSC
    )
    for (i, j, v) in zip(findnz(mat)...)
        Is[offset] = i + start_row
        Js[offset] = j + start_col
        Vs[offset] = v
        offset += 1
    end
    return offset
end

function add_I_J_V(
    offset::Int,
    Is::Vector{Int32},
    Js::Vector{Int32},
    Vs::Vector{Float64},
    start_rows::Vector{Int},
    start_cols::Vector{Int},
    mats::Vector{<: SparseMatrixCSC}
    )
    for (r, c, m) in zip(start_rows, start_cols, mats)
        offset = add_I_J_V(offset, Is, Js, Vs, r, c, m)
    end
    return offset
end

function add_hess_to_lhs()
end
