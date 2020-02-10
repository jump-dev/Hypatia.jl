#=
Copyright 2018, Chris Coey and contributors

utilities for constructing Hypatia native models and PolyJuMP.jl models
=#

module ModelUtilities

const RealOrComplex{T <: Real} = Union{T, Complex{T}}

include("domains.jl")

using LinearAlgebra
import FFTW
import Combinatorics
import GSL: sf_gamma_inc_Q
import DynamicPolynomials
const DP = DynamicPolynomials
include("interpolate.jl") # TODO remove dependence on DP

import SemialgebraicSets
const SAS = SemialgebraicSets
include("semialgebraicsets.jl")

# utilities for in-place vector and matrix rescalings for svec form
function vec_to_svec!(arr::AbstractVecOrMat{T}; rt2 = sqrt(T(2)), incr::Int = 1) where {T}
    n = size(arr, 1)
    @assert iszero(rem(n, incr))
    side = round(Int, sqrt(0.25 + 2 * div(n, incr)) - 0.5)
    k = 1
    for i in 1:side
        for j in 1:(i - 1)
            @. arr[k:(k + incr - 1), :] *= rt2
            k += incr
        end
        k += incr
    end
    return arr
end

function svec_to_vec!(arr::AbstractVecOrMat{T}; rt2 = sqrt(T(2)), incr::Int = 1) where {T}
    n = size(arr, 1)
    @assert iszero(rem(n, incr))
    side = round(Int, sqrt(0.25 + 2 * div(n, incr)) - 0.5)
    k = 1
    for i in 1:side
        for j in 1:(i - 1)
            @. arr[k:(k + incr - 1), :] /= rt2
            k += incr
        end
        k += incr
    end
    return arr
end

end
