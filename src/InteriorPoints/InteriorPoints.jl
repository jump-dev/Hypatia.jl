#=
Copyright 2018, Chris Coey and contributors

functions and caches for interior point algorithms
=#

module InteriorPoints

using Printf
using LinearAlgebra
using SparseArrays
import Hypatia.Cones
import Hypatia.Models

abstract type IPMSolver end

abstract type InteriorPoint end

include("homogselfdual.jl")

end
