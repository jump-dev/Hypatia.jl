#=
Copyright 2018, Chris Coey and contributors
=#

using Hypatia
using Test

verbose = false # test verbosity

# native interface tests
# include(joinpath(@__DIR__, "native.jl"))
# testnative(verbose, Hypatia.QRSymmCache)
# testnative(verbose, Hypatia.NaiveCache)

# MathOptInterface tests
include(joinpath(@__DIR__, "moi.jl"))
testmoi(verbose, false)
# testmoi(verbose, true) # TODO fix failure on linear1

return nothing
