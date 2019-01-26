#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia

# submodules
include("Cones/Cones.jl")
# include("LinearSystems/LinearSystems.jl")
include("Models/Models.jl")
include("InteriorPoints/InteriorPoints.jl")
include("ModelUtilities/ModelUtilities.jl")

const HYP = Hypatia
const CO = HYP.Cones
# const LS = HYP.LinearSystems
const MO = HYP.Models
const IP = HYP.InteriorPoints

# MathOptInterface
using LinearAlgebra
using SparseArrays
import MathOptInterface
const MOI = MathOptInterface
include("MathOptInterface/cones.jl")
include("MathOptInterface/wrapper.jl")

end
