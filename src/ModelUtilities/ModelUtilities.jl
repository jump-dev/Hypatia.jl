#=
Copyright 2018, Chris Coey and contributors

utilities for constructing Hypatia native models and PolyJuMP.jl models
=#

module ModelUtilities

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

# utilities for symmetric matrix scalings
function vec_to_svec!(vec::AbstractVector, rt2::Number; incr::Int = 1)
    n = length(vec)
    @assert iszero(rem(n, incr))
    side = round(Int, sqrt(0.25 + 2 * div(n, incr)) - 0.5)
    k = 1
    @inbounds for i in 1:side
        for j in 1:(i - 1)
            vec[k:(k + incr - 1)] *= rt2
            k += incr
        end
        k += incr
    end
    return vec
end

function vec_to_svec_cols!(A::AbstractMatrix, rt2::Number; incr::Int = 1)
    @inbounds for j in 1:size(A, 2)
        @views vec_to_svec!(A[:, j], rt2, incr = incr)
    end
    return A
end

end
