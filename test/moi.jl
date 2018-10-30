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
        MOI.ExponentialCone, MOI.PowerCone, MOI.GeometricMeanCone,
        MOI.PositiveSemidefiniteConeTriangle,
        MOI.LogDetConeTriangle,
    ),
    (),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    )

config = MOIT.TestConfig(
    atol = 1.2e-4,
    rtol = 1e-4,
    solve = true,
    query = true,
    modify_lhs = true,
    duals = true,
    infeas_certificates = true,
    )

unit_exclude = [
    "solve_qcp_edge_cases",
    "solve_qp_edge_cases",
    "solve_integer_edge_cases",
    "solve_objbound_edge_cases",
    ]
conic_exclude = String[
    # "lin",
    # "soc",
    # "rsoc",
    # "exp",
    # "geomean",
    # "sdp",
    # "logdet",
    # "rootdet",
    # TODO MOI bridges don't support square logdet or rootdet
    "logdets",
    "rootdets",
    ]


function testmoi(; verbose, lscachetype, usedense)
    optimizer = MOIU.CachingOptimizer(HypatiaModelData{Float64}(),
        Hypatia.Optimizer(
            verbose = verbose,
            lscachetype = lscachetype,
            usedense = usedense,
            tolrelopt = 1e-8,
            tolabsopt = 5e-8,
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

    @testset "conic tests" begin
        MOIT.contconictest(
            MOIB.SquarePSD{Float64}(
            MOIB.LogDet{Float64}( # TODO remove when MOI LogDet cone definition is fixed
            MOIB.RootDet{Float64}(
                optimizer
            ))),
            config, conic_exclude)
    end

    return nothing
end
