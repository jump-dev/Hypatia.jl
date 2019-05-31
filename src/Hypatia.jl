#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia

const HypReal = Union{AbstractFloat, Rational}
const HypRealOrComplex{T <: HypReal} = Union{T, Complex{T}}

const rt2 = sqrt(2)
const rt2i = inv(rt2)

include("linearalgebra.jl")

# submodules
include("ModelUtilities/ModelUtilities.jl")
include("Cones/Cones.jl")
include("Models/Models.jl")
include("Solvers/Solvers.jl")

# MathOptInterface
import MathOptInterface
const MOI = MathOptInterface

include("MathOptInterface/cones.jl")
include("MathOptInterface/wrapper.jl")

end
