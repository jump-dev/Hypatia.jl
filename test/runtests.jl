#=
Copyright 2018, Chris Coey and contributors
=#

using Hypatia
using Test

verbflag = false # test verbosity

# TODO interpolation tests

# native interface tests
include(joinpath(@__DIR__, "native.jl"))
testnative(verbflag)

# MathOptInterface tests
include(joinpath(@__DIR__, "moi.jl"))
testmoi(verbflag)

return nothing
