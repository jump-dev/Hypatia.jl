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
        MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval,
    ),
    (
        MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
        MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,
        # MOI.PositiveSemidefiniteConeTriangle,
        MOI.ExponentialCone,
        # MOI.PowerCone,
    ),
    (),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    )

function testmoi(verbose::Bool, usedense::Bool)
    optimizer = MOIU.CachingOptimizer(HypatiaModelData{Float64}(), Hypatia.HypatiaOptimizer(verbose=verbose, usedense=usedense))

    config = MOIT.TestConfig(
        atol = 1e-3,
        rtol = 1e-3,
        solve = true,
        query = true,
        modify_lhs = true,
        duals = true,
        infeas_certificates = true,
        )

    @testset "MathOptInterface tests" begin
    # @testset "Continuous linear problems without interval sets" begin
    #     MOIT.contlineartest(MOIB.SplitInterval{Float64}(optimizer), config)
    # end
    @testset "Continuous linear problems with interval sets" begin
        MOIT.linear10test(optimizer, config)
    end
    # @testset "Continuous conic problems" begin
    #     exclude = ["rootdet", "logdet", "sdp"] # TODO MOI does not yet support scaled PSD triangle
    #     MOIT.contconictest(
    #         MOIB.GeoMean{Float64}(
    #         # MOIB.SquarePSD{Float64}(
    #         # MOIB.LogDet{Float64}(
    #         # MOIB.RootDet{Float64}(
    #             optimizer
    #         ),#))),
    #         config, exclude)
    # end
    end

    return nothing
end
