
using Alfonso
using Test

# verbflag = true # Alfonso verbose option

# # TODO interpolation tests
#
# # load optimizer builder functions from examples folder
# egs_dir = joinpath(@__DIR__, "../examples")
# include(joinpath(egs_dir, "envelope/envelope.jl"))
# include(joinpath(egs_dir, "lp/lp.jl"))
# include(joinpath(egs_dir, "namedpoly/namedpoly.jl"))
#
# # run native and MOI interfaces on examples
# include(joinpath(@__DIR__, "nativeexamples.jl"))


import MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities

MOIU.@model(AlfonsoModelData,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan),
    (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.PositiveSemidefiniteConeTriangle, MOI.ExponentialCone, MOI.PowerCone),
    (),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    )

const optimizer = MOIU.CachingOptimizer(AlfonsoModelData{Float64}(), Alfonso.Optimizer())

const config = MOIT.TestConfig(
    atol=1e-4,
    rtol=1e-4,
    solve=true,
    query=false,
    modify_lhs=false,
    duals=false,
    infeas_certificates=false,
    )

@testset "Continuous linear problems" begin
    MOIT.contlineartest(MOIB.SplitInterval{Float64}(optimizer), config)
end

# @testset "Continuous conic problems" begin
#     exclude = ["rootdet", "logdet"]
#     MOIT.contconictest(
#         MOIB.SquarePSD{Float64}(
#         MOIB.GeoMean{Float64}(
#         MOIB.LogDet{Float64}(
#         MOIB.RootDet{Float64}(
#             optimizer
#         )))),
#         config, exclude)
# end
