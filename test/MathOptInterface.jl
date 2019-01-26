#=
Copyright 2018, Chris Coey and contributors
=#

import MathOptInterface
const MOI = MathOptInterface
MOIT = MOI.Test
MOIB = MOI.Bridges
MOIU = MOI.Utilities

MOIU.@model(HypatiaModelData,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
        MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,
        MOI.PositiveSemidefiniteConeTriangle,
        MOI.ExponentialCone, MOI.GeometricMeanCone, MOI.LogDetConeTriangle),
    (MOI.PowerCone,),
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

# function test_moi(; verbose, linear_solver, dense)
function test_moi(; verbose, dense)
    optimizer = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(HypatiaModelData{Float64}()),
        HYP.Optimizer(
            verbose = verbose,
            time_limit = 2e1,
            # linear_solver = linear_solver,
            dense = dense,
            # tol_rel_opt = 2e-8,
            # tol_abs_opt = 2e-8,
            # tol_feas = 1e-8,
            )
        )
    @testset "unit tests" begin
        MOIT.unittest(optimizer, config, unit_exclude)
    end
    @testset "linear tests" begin
        MOIT.contlineartest(optimizer, config)
    end
    @testset "conic tests" begin
        MOIT.contconictest(MOIB.SquarePSD{Float64}(MOIB.RootDet{Float64}(optimizer)), config, conic_exclude)
    end
    return
end
