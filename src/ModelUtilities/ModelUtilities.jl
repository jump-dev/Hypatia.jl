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
function vec_to_svec!(vec::AbstractVector{T}, rt2::T) where {T}
    side = round(Int, sqrt(0.25 + 2 * length(vec)) - 0.5)
    k = 0
    for i in 1:side
        for j in 1:(i - 1)
            k += 1
            vec[k] *= rt2
        end
        k += 1
    end
    return vec
end

function vec_to_svec_cols!(A::AbstractMatrix, rt2::Number)
    for j in 1:size(A, 2)
        @views vec_to_svec!(A[:, j], rt2)
    end
    return A
end

end
