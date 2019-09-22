#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

# TODO used for Pardiso - move into a Pardiso init function?
max_num_threads = length(Sys.cpu_info())
ENV["OMP_NUM_THREADS"] = max_num_threads

import Pardiso
import SparseArrays
import SparseArrays.SparseMatrixCSC
import SuiteSparse
import SuiteSparse.UMFPACK
import SuiteSparse.CHOLMOD

# TODO this will get removed when we don't care about timing inside these functions
using TimerOutputs

abstract type SparseSolverCache end

mutable struct PardisoCache <: SparseSolverCache
    use_symmetric::Bool
    analyzed::Bool
    ps::Pardiso.PardisoSolver
end
PardisoCache(use_symmetric::Bool) = PardisoCache(use_symmetric, false, Pardiso.PardisoSolver())

mutable struct UMFPACKCache <: SparseSolverCache
    analyzed::Bool
    fact::SuiteSparse.UMFPACK.UmfpackLU
    function UMFPACKCache()
        cache = new()
        cache.analyzed = false
        return cache
    end
end

mutable struct CHOLMODCache <: SparseSolverCache
    analyzed::Bool
    fact::SuiteSparse.CHOLMOD.Factor
    backup_fact::SuiteSparse.UMFPACK.UmfpackLU
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
    Pardiso.pardiso(ps, A, b)
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
    @timeit solver.timer "solve" Pardiso.pardiso(ps, x, A, b) # TODO debug
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
    Pardiso.pardiso(ps)
    return
end

release_sparse_cache(::UMFPACKCache) = nothing

release_sparse_cache(::CHOLMODCache) = nothing
