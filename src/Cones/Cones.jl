#=
Copyright 2018, Chris Coey and contributors

functions and caches for primitive cones
=#

module Cones

using LinearAlgebra
using ForwardDiff
using DiffResults

abstract type PrimitiveCone end

include("orthant.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epipersquare.jl")
include("hypoperlog.jl")
include("epiperpower.jl")
include("epipersumexp.jl")
include("hypogeomean.jl")
include("epinormspectral.jl")
include("semidefinite.jl")
include("hypoperlogdet.jl")
include("wsospolyinterp.jl")
include("wsospolyinterpmat.jl")

include("cone.jl")

export Cone

end
