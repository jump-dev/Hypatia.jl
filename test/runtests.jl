#=
Copyright 2018, Chris Coey and contributors
=#

using Hypatia
using Test

verbflag = false # test verbosity

# TODO interpolation tests

# native interface tests
# TODO test all interface functions (codecov will help)
include(joinpath(@__DIR__, "native.jl"))
testnative(verbflag, linsyscache=Hypatia.QRCholCache)
testnative(verbflag, linsyscache=Hypatia.NaiveCache)

# MathOptInterface tests
# TODO test with a variety of methods/options (eg various linsys solvers)
include(joinpath(@__DIR__, "moi.jl"))
testmoi(verbflag)

return nothing
