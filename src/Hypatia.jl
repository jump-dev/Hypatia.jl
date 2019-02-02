#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia

# submodules
include("ModelUtilities/ModelUtilities.jl")
include("Cones/Cones.jl")
include("Models/Models.jl")
include("LinearSystems/LinearSystems.jl")
include("InteriorPoints/InteriorPoints.jl")

# MathOptInterface
using LinearAlgebra
using SparseArrays
import MathOptInterface
const MOI = MathOptInterface
include("MathOptInterface/cones.jl")
include("MathOptInterface/wrapper.jl")

end
