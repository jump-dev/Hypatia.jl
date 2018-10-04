#=
Copyright 2018, Chris Coey and contributors
=#

using MathOptInterface
MOI = MathOptInterface
MOIT = MOI.Test
MOIB = MOI.Bridges
MOIU = MOI.Utilities

MOIU.@model(HypatiaModelData,
    (),
    (
        MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,
        # MOI.Interval,
    ),
    (
        MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
        MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,
        MOI.PositiveSemidefiniteConeTriangle,
        MOI.ExponentialCone,
        # MOI.PowerCone,
    ),
    (),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    )

optimizer = MOIU.CachingOptimizer(HypatiaModelData{Float64}(), Hypatia.HypatiaOptimizer())

config = MOIT.TestConfig(
    atol=1e-3,
    rtol=1e-3,
    solve=true,
    query=true,
    modify_lhs=true,
    duals=true,
    infeas_certificates=true,
    )

function testmoi(verbflag::Bool)
    @testset "MathOptInterface tests" begin
    @testset "Continuous linear problems" begin
        MOIT.contlineartest(
            MOIB.SplitInterval{Float64}(
                optimizer
            ),
            config)
    end
    @testset "Continuous conic problems" begin
        exclude = ["rootdet", "logdet", "sdp"] # TODO bridges not working? should not need to exclude in future
        MOIT.contconictest(
            # MOIB.SquarePSD{Float64}(
            MOIB.GeoMean{Float64}(
            # MOIB.LogDet{Float64}(
            # MOIB.RootDet{Float64}(
                optimizer
            ),#))),
            config, exclude)
    end
    end
    return nothing
end
