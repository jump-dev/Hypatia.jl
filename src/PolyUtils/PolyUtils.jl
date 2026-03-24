#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

"""
Utilities for constructing interpolant-basis polynomial sum-of-squares models.
"""
module PolyUtils

using DocStringExtensions

using LinearAlgebra

import Combinatorics
import SpecialFunctions: gamma_inc

include("realdomains.jl")
include("realinterp.jl")
include("complex.jl")

end
