#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
utilities for HSL
only works with Float64 and Float32
=#

module HSLExt

import HSL
import Hypatia
import SparseArrays.SparseMatrixCSC
import LinearAlgebra.BlasReal

mutable struct HSLSymCache{T <: BlasReal} <: Hypatia.SparseSymCache{T}
    analyzed::Bool
    ma57::HSL.Ma57{T}
    work::Vector{T}
    function HSLSymCache{T}() where {T <: BlasReal}
        cache = new{T}()
        cache.analyzed = false
        return cache
    end
end

Hypatia.int_type(::HSLSymCache) = Int

function Hypatia.update_fact(
    cache::HSLSymCache{T},
    A::SparseMatrixCSC{T, Int},
) where {T <: BlasReal}
    if !cache.analyzed
        cache.ma57 = HSL.Ma57(A)
        cache.work = Vector{T}(undef, cache.ma57.n)
        cache.analyzed = true
    else
        copyto!(cache.ma57.vals, A.nzval)
    end
    HSL.ma57_factorize!(cache.ma57)
    return
end

function Hypatia.inv_prod(
    cache::HSLSymCache{T},
    x::Vector{T},
    A::SparseMatrixCSC{T, Int},
    b::Vector{T},
) where {T <: BlasReal}
    # MA57 only has the option to take iterative refinement steps for a single-column RHS
    copyto!(x, b)
    HSL.ma57_solve!(cache.ma57, x, cache.work)

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

Hypatia.free_memory(cache::HSLSymCache) = nothing

end #module
