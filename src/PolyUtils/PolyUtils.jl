"""
Utilities for constructing interpolant-basis polynomial sum-of-squares models.
"""
module PolyUtils

using DocStringExtensions

using LinearAlgebra
# delete later, affects qr. see https://github.com/JuliaLang/julia/pull/40623
if VERSION < v"1.7.0-DEV.1188"
    const ColumnNorm = Val{true}
end

import Combinatorics
import SpecialFunctions: gamma_inc

include("realdomains.jl")
include("realinterp.jl")
include("complex.jl")

end
