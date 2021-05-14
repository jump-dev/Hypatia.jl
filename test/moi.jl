#=
MOI.Test linear and conic tests
=#

using Test
import MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIU = MOI.Utilities
import Hypatia

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
    # "sdp",
    # "norminf",
    # "normone",
    # "soc",
    # "rsoc",
    # "normspec",
    # "normnuc",
    # "pow",
    # "dualpow",
    # "geomean",
    # "rootdet",
    "rootdets",
    # "exp",
    # "dualexp",
    # "logdet",
    "logdets",
    # "relentr",
    ]

function test_moi(T::Type{<:Real}; solver_options...)
    optimizer = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(MOIU.Model{T}()),
        Hypatia.Optimizer{T}(; solver_options...)
        )

    tol = 2sqrt(sqrt(eps(T)))
    config = MOIT.TestConfig{T}(
        atol = tol,
        rtol = tol,
        solve = true,
        query = true,
        modify_lhs = true,
        duals = true,
        infeas_certificates = true,
        )

    @testset "linear tests" begin
        MOIT.contlineartest(optimizer, config)
    end

    if T == Float64
        # test other real types, waiting for https://github.com/jump-dev/MathOptInterface.jl/issues/841
        @testset "unit tests" begin
            MOIT.unittest(optimizer, config, unit_exclude)
        end
        @testset "conic tests" begin
            bridged = MOI.Bridges.Constraint.Square{T}(optimizer)
            MOIT.contconictest(bridged, config, conic_exclude)
        end
    end

    return
end
