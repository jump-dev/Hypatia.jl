#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

This module uses code from https://github.com/JuliaOpt/ConicBenchmarkUtilities.jl (thanks to Miles Lubin and contributors).

The pupose of this module is to convert CBF files into native intput for Hypatia, and should be replaced by ConicBenchmarkUtilities.jl when updated for MOI.
=#
module Translate
    using GZip
    using SparseArrays
    using Hypatia
    using LinearAlgebra

    include("cbf_input.jl")
    include("mpb.jl")
    include("hypatia.jl")
end
