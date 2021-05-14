#=
helpers for sparse factorizations and linear solves
=#

import SparseArrays.SparseMatrixCSC
import SuiteSparse
const SuiteSparseInt = SuiteSparse.CHOLMOD.SuiteSparse_long

#=
nonsymmetric
=#

abstract type SparseNonSymCache{T <: Real} end

# only works with Float64
mutable struct UMFPACKNonSymCache{Float64} <: SparseNonSymCache{Float64}
    analyzed::Bool
    umfpack::SuiteSparse.UMFPACK.UmfpackLU
    function UMFPACKNonSymCache{Float64}()
        cache = new{Float64}()
        cache.analyzed = false
        return cache
    end
end

# restrict int type to SuiteSparse_long
int_type(::UMFPACKNonSymCache) = SuiteSparseInt

function update_fact(
    cache::UMFPACKNonSymCache{Float64},
    A::SparseMatrixCSC{Float64, SuiteSparseInt},
    )
    if !cache.analyzed
        cache.umfpack = lu(A) # symbolic and numeric factorization
        cache.analyzed = true
    else
        lu!(cache.umfpack, A, check = false)
    end
    return
end

function inv_prod(
    cache::UMFPACKNonSymCache{Float64},
    x::Vector{Float64},
    A::SparseMatrixCSC{Float64, SuiteSparseInt},
    b::Vector{Float64},
    )
    ldiv!(x, cache.umfpack, b) # does not repeat symbolic or numeric factorization
    return x
end

# default to UMFPACK
SparseNonSymCache{Float64}() = UMFPACKNonSymCache{Float64}()

#=
symmetric
=#

abstract type SparseSymCache{T <: Real} end

# helper for symmetric solvers that need nonzero diagonal
diag_min(::SparseSymCache{T}) where T = zero(T)

# only works with Float64
mutable struct CHOLMODSymCache{Float64} <: SparseSymCache{Float64}
    analyzed::Bool
    cholmod::SuiteSparse.CHOLMOD.Factor
    function CHOLMODSymCache{Float64}()
        cache = new{Float64}()
        cache.analyzed = false
        return cache
    end
end
int_type(::CHOLMODSymCache) = SuiteSparseInt

# CHOLMOD needs nonzero diagonal
diag_min(::SparseSymCache{Float64}) = sqrt(eps())

function update_fact(
    cache::CHOLMODSymCache{Float64},
    A::SparseMatrixCSC{Float64, SuiteSparseInt},
    )
    A_symm = Symmetric(A, :L)

    if !cache.analyzed
        cache.cholmod = SuiteSparse.CHOLMOD.ldlt(A_symm, check = false)
        cache.analyzed = true
    else
        ldlt!(cache.cholmod, A_symm, check = false)
    end

    if !issuccess(cache.cholmod)
        @warn("numerical failure: sparse factorization failed")
        ldlt!(cache.cholmod, A_symm, shift = 1e-4, check = false)
        if !issuccess(cache.cholmod)
            @warn("numerical failure: could not fix sparse factorization failure")
        end
    end
    return
end

function inv_prod(
    cache::CHOLMODSymCache{Float64},
    x::Vector{Float64},
    A::SparseMatrixCSC{Float64, SuiteSparseInt},
    b::Vector{Float64},
    )
    x .= cache.cholmod \ b # TODO try to make this in-place
    return x
end

# default to CHOLMOD
SparseSymCache{Float64}() = CHOLMODSymCache{Float64}()

#=
helpers
=#

free_memory(::Union{
    UMFPACKNonSymCache{Float64},
    CHOLMODSymCache{Float64},
    }) = nothing
