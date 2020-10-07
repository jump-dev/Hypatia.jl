#=
utilities for constructing Hypatia native models and PolyJuMP.jl models
=#

module ModelUtilities

const RealOrComplex{T <: Real} = Union{T, Complex{T}}

include("domains.jl")

using LinearAlgebra
import Combinatorics
include("interpolate.jl")

import DynamicPolynomials
const DP = DynamicPolynomials
include("polynomials.jl") # TODO possibly remove these functions, then remove dependence on DP (from Project.toml also)

# utilities for in-place vector and matrix rescalings for svec form
function vec_to_svec!(arr::AbstractVecOrMat{T}; rt2 = sqrt(T(2)), incr::Int = 1) where {T}
    n = size(arr, 1)
    @assert iszero(rem(n, incr))
    side = round(Int, sqrt(0.25 + 2 * div(n, incr)) - 0.5)
    k = 1
    for i in 1:side
        for j in 1:(i - 1)
            @. @views arr[k:(k + incr - 1), :] *= rt2
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

function eval_lagrange_polys(F, points, state_lb::Vector{T}, state_ub::Vector{T}, order, shift) where {T}
    points_shift = similar(points)
    if shift
        for i in 1:size(points, 1)
            points_shift[i, :] = (points[i, :] .- (state_lb .+ state_ub) ./ 2) ./ (state_ub .- state_lb) .* 2
        end
    end
    X = hcat(points_shift) # 1 by 1 matrix
    V = make_chebyshev_vandermonde(X, 2order)
    return F \ V'
end

function initial_wsos_point(F, points, state_lb::Vector{T}, state_ub::Vector{T}, order, shift) where {T}
    point_evals = eval_lagrange_polys(F, points, state_lb, state_ub, order, shift)
    return vec(sum(point_evals, dims = 2))
end

end
