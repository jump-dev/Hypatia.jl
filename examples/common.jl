#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for examples
=#

using Test
import Random
using LinearAlgebra
import LinearAlgebra.BlasReal

import Hypatia
import Hypatia.ModelUtilities
import Hypatia.Cones
import Hypatia.Models
import Hypatia.Solvers

abstract type InstanceSet end
struct MinimalInstances <: InstanceSet end
struct FastInstances <: InstanceSet end
struct SlowInstances <: InstanceSet end
struct LinearOperatorsInstances <: InstanceSet end

abstract type ExampleInstance{T <: Real} end

example_tests(::Type{<:ExampleInstance}, ::InstanceSet) = Tuple[]

# NOTE this is a workaround for randn's lack of support for BigFloat
Random.randn(R::Type{BigFloat}, dims::Vararg{Int, N} where N) = R.(randn(dims...))
Random.randn(R::Type{Complex{BigFloat}}, dims::Vararg{Int, N} where N) = R.(randn(ComplexF64, dims...))

# helper for calculating solution violations
function relative_residual(residual::Vector{T}, constant::Vector{T}) where {T <: Real}
    @assert length(residual) == length(constant)
    return T[residual[i] / max(one(T), constant[i]) for i in eachindex(constant)]
end
