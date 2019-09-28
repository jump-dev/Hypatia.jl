#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia

const RealOrComplex{T <: Real} = Union{T, Complex{T}}

using LinearAlgebra

include("linearalgebra/blockmatrix.jl")
include("linearalgebra/dense.jl")
include("linearalgebra/sparse.jl")

import Requires
function __init__()
    Requires.@require Pardiso = "46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" include(joinpath(@__DIR__(), "linearalgebra", "Pardiso.jl"))
end


# submodules
include("ModelUtilities/ModelUtilities.jl")
include("Cones/Cones.jl")
include("Models/Models.jl")
include("Solvers/Solvers.jl")

# MathOptInterface
using SparseArrays
import MathOptInterface
const MOI = MathOptInterface

include("MathOptInterface/cones.jl")
include("MathOptInterface/wrapper.jl")

end
