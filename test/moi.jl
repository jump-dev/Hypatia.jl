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

function testmoi(; verbose, lscachetype, usedense)
    optimizer = MOIU.CachingOptimizer(HypatiaModelData{Float64}(), Hypatia.HypatiaOptimizer(
        verbose = verbose,
        lscachetype = lscachetype,
        usedense = usedense,
        tolrelopt = 1e-8,
        tolabsopt = 1e-8,
        tolfeas = 1e-7,
        ))

    config = MOIT.TestConfig(
        atol = 1e-4,
        rtol = 1e-4,
        solve = true,
        query = true,
        modify_lhs = true,
        duals = true,
        infeas_certificates = true,
        )

    MOIT.contlineartest(MOIB.SplitInterval{Float64}(optimizer), config)

    MOIT.linear10test(optimizer, config)

    exclude = ["rootdet", "logdet", "sdp"] # TODO MOI does not yet support scaled PSD triangle
    MOIT.contconictest(
        MOIB.GeoMean{Float64}(
        # MOIB.SquarePSD{Float64}(
        # MOIB.LogDet{Float64}(
        # MOIB.RootDet{Float64}(
            optimizer
        ),#))),
        config, exclude)

    return nothing
end
