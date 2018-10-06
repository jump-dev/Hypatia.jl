#=
Copyright 2018, Chris Coey and contributors
=#

using Hypatia
using Test

verbose = false # test verbosity

# native interface tests
include(joinpath(@__DIR__, "native.jl"))
testnative(verbose, Hypatia.QRCholCache)
# testnative(verbose, Hypatia.QRConjGradCache) # TODO fails for some problems
testnative(verbose, Hypatia.NaiveCache)

# MathOptInterface tests
include(joinpath(@__DIR__, "moi.jl"))
testmoi(verbose, true)
# testmoi(verbose, false) # TODO fails on empty sparse A


return nothing
