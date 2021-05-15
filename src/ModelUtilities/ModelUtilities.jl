#=
utilities for constructing Hypatia native models and PolyJuMP.jl models
=#

module ModelUtilities

include("domains.jl")

using LinearAlgebra
import Combinatorics
include("interpolate.jl")
include("complex.jl")

end
