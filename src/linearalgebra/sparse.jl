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

function update_sparse_fact(cache::UMFPACKNonSymCache, A::SparseMatrixCSC{Float64, <:Integer})
    if !cache.analyzed
        cache.umfpack = lu(A) # symbolic and numeric factorization
        cache.analyzed = true
    else
        # TODO this is a hack around lack of interface https://github.com/JuliaLang/julia/issues/33323
        # update nzval field in the factorization
        copyto!(cache.umfpack.nzval, A.nzval)
        # do not indicate that the numeric factorization has been computed
        cache.umfpack.numeric = C_NULL
        SuiteSparse.UMFPACK.umfpack_numeric!(cache.umfpack) # will only repeat numeric factorization
    end
    return
end

function solve_sparse_system(cache::UMFPACKNonSymCache, x::Matrix{Float64}, A::SparseMatrixCSC{Float64, <:Integer}, b::Matrix{Float64})
    ldiv!(x, cache.umfpack, b) # will not repeat factorizations
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

mutable struct PardisoSymCache{T <: Real} <: SparseSymCache{T}
    analyzed::Bool
    pardiso::Pardiso.PardisoSolver
    diag_pert::Float64
    int_type::Type{Int32}
    function PardisoSymCache{Float64}(; diag_pert = NaN)
        cache = new{Float64}()
        cache.analyzed = false
        cache.pardiso = Pardiso.PardisoSolver()
        Pardiso.set_matrixtype!(cache.pardiso, Pardiso.REAL_SYM_INDEF) # tell Pardiso the matrix is symmetric indefinite
        cache.diag_pert = diag_pert
        cache.int_type = Int32
        return cache
    end
end
PardisoSymCache{T}() where {T <: Real} = error("Pardiso only works with real type Float64")
PardisoSymCache(; diag_pert = NaN) = PardisoSymCache{Float64}(diag_pert = diag_pert)

mutable struct CHOLMODSymCache{T <: Real} <: SparseSymCache{T}
    analyzed::Bool
    cholmod::SuiteSparse.CHOLMOD.Factor
    diag_pert::Float64
    int_type::Type{<: Integer}
    function CHOLMODSymCache{Float64}(; diag_pert = NaN)
        cache = new{Float64}()
        cache.analyzed = false
        cache.diag_pert = diag_pert
        cache.int_type = SuiteSparse.CHOLMOD.SuiteSparse_long
        return cache
    end
end
CHOLMODSymCache{T}() where {T <: Real} = error("CHOLMOD only works with real type Float64")
CHOLMODSymCache() = CHOLMODSymCache{Float64}()

function update_sparse_fact(cache::CHOLMODSymCache, A::SparseMatrixCSC{Float64, <:Integer})
    A_symm = Symmetric(A, :L)
    if !cache.analyzed
        cache.cholmod = SuiteSparse.CHOLMOD.ldlt(A_symm, check = false)
        cache.analyzed = true
    else
        ldlt!(cache.cholmod, A_symm, check = true)
    end
    if !issuccess(cache.cholmod)
        # @warn("numerical failure: sparse factorization failed")
        # ldlt!(cache.cholmod, A_symm, shift = 1e-4, check = false)
        # if !issuccess(cache.cholmod)
        #     @warn("numerical failure: sparse factorization failed again")
        #     ldlt!(cache.cholmod, A_symm, shift = 1e-8 * maximum(abs, A[j, j] for j in 1:size(A_symm, 1)), check = false)
        #     if !issuccess(cache.cholmod)
        #         @warn("numerical failure: could not fix sparse factorization failure")
        #     end
        # end
    end
    return
end

function solve_sparse_system(cache::CHOLMODSymCache, x::Matrix{Float64}, A::SparseMatrixCSC{Float64, <:Integer}, b::Matrix{Float64})
    x .= cache.cholmod \ b
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

function update_sparse_fact(cache::PardisoSparseCache, A::SparseMatrixCSC{Float64, Int32})
    pardiso = cache.pardiso

    if !cache.analyzed
        Pardiso.pardisoinit(pardiso)
        # don't ignore other iparms
        Pardiso.set_iparm!(pardiso, 1, 1)
        # solve transposed problem (Pardiso accepts CSR matrices)
        Pardiso.set_iparm!(pardiso, 12, 1)
        # perturbation for small pivots (default 8 for symmetric, 13 for nonsymmetric)
        if Pardiso.get_matrixtype(pardiso) == Pardiso.REAL_SYM_INDEF
            Pardiso.set_iparm!(pardiso, 10, 8)
        end
        # maximum number of iterative refinement steps (default = 2)
        Pardiso.set_iparm!(pardiso, 8, 5)
        Pardiso.set_phase!(pardiso, Pardiso.ANALYSIS)
        Pardiso.pardiso(pardiso, A, Float64[])
        cache.analyzed = true
    end

    Pardiso.set_phase!(pardiso, Pardiso.NUM_FACT)
    Pardiso.pardiso(pardiso, A, Float64[])

    return
end

function solve_sparse_system(cache::PardisoSparseCache, x::Matrix{Float64}, A::SparseMatrixCSC{Float64, Int32}, b::Matrix{Float64})
    pardiso = cache.pardiso

    Pardiso.set_phase!(pardiso, Pardiso.SOLVE_ITERATIVE_REFINE)
    Pardiso.pardiso(pardiso, x, A, b)
    return x
end

function free_memory(cache::PardisoSparseCache)
    Pardiso.set_phase!(cache.pardiso, Pardiso.RELEASE_ALL)
    Pardiso.pardiso(cache.pardiso)
    return
end
free_memory(::SuiteSparseSparseCache) = nothing
