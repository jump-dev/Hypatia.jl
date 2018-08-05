
using Alfonso
using Test
using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities


# TODO interpolation tests

# load optimizer builder functions from examples folder
egs_dir = joinpath(@__DIR__, "../examples")
include(joinpath(egs_dir, "envelope/envelope.jl"))
include(joinpath(egs_dir, "lp/lp.jl"))
include(joinpath(egs_dir, "namedpoly/namedpoly.jl"))

# run native and MOI interfaces on examples
include(joinpath(@__DIR__, "nativeexamples.jl"))
# include(joinpath(@__DIR__, "moiexamples.jl"))

# run MOI continuous linear and conic tests
# MOIU.@model AlfonsoModelData () () (Zeros, Nonnegatives) () () () (VectorOfVariables,) (VectorAffineFunction,)
# const optimizer = MOIU.CachingOptimizer(AlfonsoModelData{Float64}(), Alfonso.Optimizer())
# const config = MOIT.TestConfig(atol=1e-4, rtol=1e-4)

# TODO use bridges e.g. GreaterThan constraints into Nonnegatives constraints
# @testset "Continuous linear problems" begin
#     MOIT.contlineartest(MOIB.SplitInterval{Float64}(optimizer), config)
# end
