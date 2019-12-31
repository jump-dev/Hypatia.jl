#=
Copyright 2018, Chris Coey and contributors
=#

using Test
import MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities
import Hypatia
const SO = Hypatia.Solvers

config = MOIT.TestConfig(
    atol = 2e-4,
    rtol = 2e-4,
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
    "solve_zero_one_with_bounds_1",
    "solve_zero_one_with_bounds_2",
    "solve_zero_one_with_bounds_3",
    "solve_unbounded_model", # dual equalities are inconsistent, so detect dual infeasibility but currently no certificate or status
    "number_threads", # no way to set threads currently
    "solve_result_index", # TODO only get one result from Hypatia
    ]

conic_exclude = String[
    # "lin",
    # "norminf",
    # "normone",
    # "soc",
    # "rsoc",
    # "exp",
    # "dualexp",
    # "pow",
    # "dualpow",
    # "geomean",
    # "sdp",
    # "logdet",
    # "rootdet",
    # square logdet and rootdet cones not handled
    "logdets",
    "rootdets",
    ]

function test_moi(T::Type{<:Real}; options...)
    optimizer = MOIU.CachingOptimizer(MOIU.UniversalFallback(MOIU.Model{T}()), Hypatia.Optimizer{T}(; options...))

    @testset "unit tests" begin
        MOIT.unittest(optimizer, config, unit_exclude)
    end

    @testset "linear tests" begin
        MOIT.contlineartest(optimizer, config)
    end

    @testset "conic tests" begin
        MOIT.contconictest(MOIB.Constraint.Square{T}(optimizer), config, conic_exclude)
    end

    return
end
