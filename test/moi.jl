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

config = MOIT.TestConfig(
    atol = 1e-4,
    rtol = 1e-4,
    solve = true,
    query = true,
    modify_lhs = true,
    duals = true,
    infeas_certificates = true,
    )

unit_exclude = [
    "solve_blank_obj", # TODO fix this?
    "solve_qcp_edge_cases",
    "solve_qp_edge_cases",
    "solve_integer_edge_cases",
    "solve_objbound_edge_cases",
    ]
conic_exclude = [
    "rootdet",
    "logdet",
    "sdp",
    ]


function testmoi(; verbose, lscachetype, usedense)
    optimizer = MOIU.CachingOptimizer(HypatiaModelData{Float64}(),
        Hypatia.Optimizer(
            verbose = verbose,
            lscachetype = lscachetype,
            usedense = usedense,
            tolrelopt = 1e-8,
            tolabsopt = 1e-8,
            tolfeas = 1e-7,
            )
        )
    @testset "unit tests" begin
        MOIT.unittest(MOIB.SplitInterval{Float64}(optimizer), config, unit_exclude)
        MOIT.unittest(optimizer, config, unit_exclude)
    end

    @testset "linear tests" begin
        MOIT.contlineartest(MOIB.SplitInterval{Float64}(optimizer), config)
        MOIT.linear10test(optimizer, config)
    end

    # TODO MOI does not yet support scaled PSD triangle
    @testset "conic tests" begin
        MOIT.contconictest(
            MOIB.GeoMean{Float64}(
            # MOIB.SquarePSD{Float64}(
            # MOIB.LogDet{Float64}(
            # MOIB.RootDet{Float64}(
                optimizer
            ),#))),
            config, conic_exclude)
    end

    return nothing
end
