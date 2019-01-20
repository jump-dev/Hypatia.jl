#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia

using Printf
using LinearAlgebra
using SparseArrays

# submodules
include("Cones/Cones.jl")
include("LinearSystems/LinearSystems.jl")
include("ModelUtilities/ModelUtilities.jl")

# core
include("preprocess.jl")
include("model.jl")
include("solve.jl")

# MathOptInterface
import MathOptInterface
const MOI = MathOptInterface
include("MathOptInterface/MOI_cones.jl")
include("MathOptInterface/MOI_wrapper.jl")

end
