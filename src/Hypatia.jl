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

# MathOptInterface
# import MathOptInterface
# const MOI = MathOptInterface
# include("MathOptInterface/MOI_cones.jl")
# include("MathOptInterface/MOI_wrapper.jl")

end
