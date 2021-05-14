#=
utilities for HSL
only works with Float64 and Float32
=#

import LinearAlgebra.BlasReal

mutable struct HSLSymCache{T <: BlasReal} <: SparseSymCache{T}
    analyzed::Bool
    ma57::HSL.Ma57{T}
    function HSLSymCache{T}() where {T <: BlasReal}
        cache = new{T}()
        cache.analyzed = false
        return cache
    end
end

int_type(::HSLSymCache) = Int

function update_fact(
    cache::HSLSymCache{T},
    A::SparseMatrixCSC{T, Int}
    ) where {T <: BlasReal}
    if !cache.analyzed
        cache.ma57 = HSL.Ma57(A)
        cache.analyzed = true
    else
        copyto!(cache.ma57.vals, A.nzval)
    end
    HSL.ma57_factorize(cache.ma57)
    return
end

function inv_prod(
    cache::HSLSymCache{T},
    x::Vector{T},
    A::SparseMatrixCSC{T, Int},
    b::Vector{T},
    ) where {T <: BlasReal}
    # MA57 only has the option to take iterative refinement steps for a single-column RHS
    ma57 = cache.ma57
    copyto!(x, b)
    HSL.ma57_solve!(ma57, x)

    # info we could print
    # ma57.info.backward_error1
    # ma57.info.backward_error2
    # ma57.info.matrix_inf_norm
    # ma57.info.scaled_residuals
    # ma57.info.cond1
    # ma57.info.cond2
    # ma57.info.error_inf_norm

    return x
end

free_memory(cache::HSLSymCache) = nothing
