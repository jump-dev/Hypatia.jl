#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia

const RealOrComplex{T <: Real} = Union{T, Complex{T}}

using LinearAlgebra

include("linearalgebra/blockmatrix.jl")
include("linearalgebra/dense.jl")
include("linearalgebra/sparse.jl")
include("linearalgebra/Pardiso.jl")

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
