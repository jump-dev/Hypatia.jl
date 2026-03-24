#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

"""
A Julia package for generic conic optimization with interior point algorithms.
"""
module Hypatia

using DocStringExtensions
const RealOrComplex{T <: Real} = Union{T, Complex{T}}

# linear algebra helpers
using LinearAlgebra
using GenericLinearAlgebra
include("linearalgebra/dense.jl")
include("linearalgebra/sparse.jl")

# submodules
include("PolyUtils/PolyUtils.jl")
include("Cones/Cones.jl")
include("Models/Models.jl")
include("Solvers/Solvers.jl")

# MathOptInterface helpers
using SparseArrays
import MathOptInterface
const MOI = MathOptInterface
const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction
const VV = MOI.VectorOfVariables
const VAF = MOI.VectorAffineFunction
import LowRankOpt as LRO
include("MathOptInterface/cones.jl")
include("MathOptInterface/transform.jl")
include("MathOptInterface/wrapper.jl")

end
