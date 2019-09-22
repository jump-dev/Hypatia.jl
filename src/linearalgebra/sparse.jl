#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

helpers for sparse factorizations and linear solves

TODO handle conditional dependencies / glue code, see https://github.com/JuliaLang/Pkg.jl/issues/1285
=#

# TODO used for Pardiso - move into a Pardiso init function?
ENV["OMP_NUM_THREADS"] = length(Sys.cpu_info())

import SparseArrays.SparseMatrixCSC
import Pardiso # TODO make optional
import SuiteSparse

#=
nonsymmetric
=#

abstract type SparseNonSymCache{T <: Real} end

reset_sparse_cache(cache::SparseNonSymCache) = (cache.analyzed = false)

mutable struct PardisoNonSymCache{T <: Real} <: SparseNonSymCache{T}
    analyzed::Bool
    pardiso::Pardiso.PardisoSolver
    function PardisoNonSymCache{Float64}()
        cache = new{Float64}()
        cache.analyzed = false
        cache.pardiso = Pardiso.PardisoSolver()
        Pardiso.set_matrixtype!(cache.pardiso, Pardiso.REAL_NONSYM) # tell Pardiso the matrix is nonsymmetric
        return cache
    end
end
PardisoNonSymCache{T}() where {T <: Real} = error("Pardiso only works with real type Float64")
PardisoNonSymCache() = PardisoNonSymCache{Float64}()

mutable struct UMFPACKNonSymCache{T <: Real} <: SparseNonSymCache{T}
    analyzed::Bool
    umfpack::SuiteSparse.UMFPACK.UmfpackLU
    function UMFPACKNonSymCache{Float64}()
        cache = new{Float64}()
        cache.analyzed = false
        return cache
    end
end
UMFPACKNonSymCache{T}() where {T <: Real} = error("UMFPACK only works with real type Float64")
UMFPACKNonSymCache() = UMFPACKNonSymCache{Float64}()

function solve_sparse_system(cache::UMFPACKNonSymCache, x::Matrix{Float64}, A::SparseMatrixCSC{Float64, <:Integer}, b::Matrix{Float64})
    if !cache.analyzed
        cache.umfpack = lu(A)
    end

    umfpack = cache.umfpack
    # TODO this is a hack around lack of interface https://github.com/JuliaLang/julia/issues/33323
    copyto!(umfpack.nzval, A.nzval)
    umfpack.numeric = C_NULL
    ldiv!(x, umfpack, b)

    return x
end

# default to UMFPACK
SparseNonSymCache{Float64}() = UMFPACKNonSymCache{Float64}()
SparseNonSymCache{T}() where {T <: Real} = error("Sparse caches only work with real type Float64")
SparseNonSymCache() = SparseNonSymCache{Float64}()

#=
symmetric
=#

abstract type SparseSymCache{T <: Real} end

reset_sparse_cache(cache::SparseSymCache) = (cache.analyzed = false)

mutable struct PardisoSymCache{T <: Real} <: SparseSymCache{T}
    analyzed::Bool
    pardiso::Pardiso.PardisoSolver
    function PardisoSymCache{Float64}()
        cache = new{Float64}()
        cache.analyzed = false
        cache.pardiso = Pardiso.PardisoSolver()
        Pardiso.set_matrixtype!(cache.pardiso, Pardiso.REAL_SYM_INDEF) # tell Pardiso the matrix is symmetric indefinite
        return cache
    end
end
PardisoSymCache{T}() where {T <: Real} = error("Pardiso only works with real type Float64")
PardisoSymCache() = PardisoSymCache{Float64}()

mutable struct CHOLMODSymCache{T <: Real} <: SparseSymCache{T}
    analyzed::Bool
    cholmod::SuiteSparse.CHOLMOD.Factor
    function CHOLMODSymCache{Float64}()
        cache = new{Float64}()
        cache.analyzed = false
        return cache
    end
end
CHOLMODSymCache{T}() where {T <: Real} = error("CHOLMOD only works with real type Float64")
CHOLMODSymCache() = CHOLMODSymCache{Float64}()

function solve_sparse_system(cache::CHOLMODSymCache, x::Matrix{Float64}, A::SparseMatrixCSC{Float64, <:Integer}, b::Matrix{Float64})
    if !cache.analyzed
        cache.cholmod = SuiteSparse.CHOLMOD.ldlt(Symmetric(A, :L), check = false) # TODO do we need check = false? and the symmetric wrapper?
    end

    cholmod = cache.cholmod
    A_symm = Symmetric(A, :L)
    ldlt!(cholmod, A_symm, check = false)
    if !issuccess(cholmod)
        ldlt!(cholmod, A_symm, shift = sqrt(eps(Float64)), check = false)
        if !issuccess(cholmod)
            @warn("numerical failure: could not fix sparse factorization failure")
        end
    end
    x .= cholmod \ b

    return x
end

# default to CHOLMOD
SparseSymCache{Float64}() = CHOLMODSymCache{Float64}()
SparseSymCache{T}() where {T <: Real} = error("Sparse caches only work with real type Float64")
SparseSymCache() = SparseSymCache{Float64}()

#=
helpers
=#

PardisoSparseCache = Union{PardisoSymCache{Float64}, PardisoNonSymCache{Float64}}
SuiteSparseSparseCache = Union{UMFPACKNonSymCache{Float64}, CHOLMODSymCache{Float64}}

function solve_sparse_system(cache::PardisoSparseCache, x::Matrix{Float64}, A::SparseMatrixCSC{Float64, Int32}, b::Matrix{Float64})
    pardiso = cache.pardiso

    if !cache.analyzed
        # TODO comment what these lines do
        Pardiso.pardisoinit(pardiso)
        Pardiso.set_iparm!(pardiso, 1, 1)
        Pardiso.set_iparm!(pardiso, 12, 1)
        Pardiso.set_phase!(pardiso, Pardiso.ANALYSIS)
        # LinearAlgebra.copytri!(A, 'L') # TODO should not be needed
        Pardiso.pardiso(pardiso, A, b)
    end

    Pardiso.set_phase!(pardiso, Pardiso.NUM_FACT_SOLVE_REFINE) # TODO can this be moved up so it is only set once?
    Pardiso.pardiso(pardiso, x, A, b) # TODO debug

    return x
end

function free_memory(cache::PardisoSparseCache)
    Pardiso.set_phase!(cache.pardiso, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(cache.pardiso)
    return
end
free_memory(::SuiteSparseSparseCache) = nothing
