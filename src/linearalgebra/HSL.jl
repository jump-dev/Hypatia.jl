#=
utilities for HSL 
=#

import LinearAlgebra.BlasReal

mutable struct HSLSymCache{T <: BlasReal} <: SparseSymCache{T}
    analyzed::Bool
    ma57::HSL.Ma57{T}
    diag_pert
    function HSLSymCache{T}(; diag_pert = zero(T)) where {T <: BlasReal}
        cache = new{T}()
        cache.analyzed = false
        cache.diag_pert = diag_pert
        return cache
    end
end
HSLSymCache{T}() where {T <: Real} = error("HSLSymCache only works with real type Float64 or Float32")
HSLSymCache() = HSLSymCache{Float64}()

int_type(::HSLSymCache) = Int

function update_fact(cache::HSLSymCache, A::SparseMatrixCSC{<:HSL.Ma57Data, Int})
    if !cache.analyzed
        cache.ma57 = HSL.Ma57(A)
        cache.analyzed = true
    else
        copyto!(cache.ma57.vals, A.nzval)
    end
    HSL.ma57_factorize(cache.ma57)
    return
end

function inv_prod(cache::HSLSymCache, x::Vector{T}, A::SparseMatrixCSC{<:HSL.Ma57Data, Int}, b::Vector{T}) where {T <: BlasReal}
    # NOTE MA57 only has the option to take iterative refinement steps for a single-column RHS
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
