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
    "lin",
    "norminf",
    "normone",
    "soc",
    "rsoc",
    "exp",
    "dualexp",
    "pow",
    "dualpow",
    # "geomean",
    "relentr",
    "normspec",
    "normnuc",
    "sdp",
    "logdet", # TODO currently failing due to barrier with high parameter
    "rootdet",
    # square logdet and rootdet cones not handled
    "logdets",
    "rootdets",
    ]

function test_moi(T::Type{<:Real}; solver_options...)
    optimizer = MOIU.CachingOptimizer(MOIU.UniversalFallback(MOIU.Model{T}()), Hypatia.Optimizer{T}(; solver_options...))

    tol = sqrt(sqrt(Float64(eps(T)))) # TODO remove Float64, waiting for MOI to be tagged after https://github.com/jump-dev/MathOptInterface.jl/pull/1176
    config = MOIT.TestConfig{T}(
        atol = tol,
        rtol = tol,
        solve = true,
        query = true,
        modify_lhs = true,
        duals = true,
        infeas_certificates = true,
        )

    # @testset "linear tests" begin
    #     MOIT.contlineartest(optimizer, config)
    # end

    if T == Float64
        # NOTE test other real types, waiting for https://github.com/jump-dev/MathOptInterface.jl/issues/841
        # @testset "unit tests" begin
        #     MOIT.unittest(optimizer, config, unit_exclude)
        # end
        @testset "conic tests" begin
            @info("no bridges")
            MOIT.contconictest(optimizer, config, conic_exclude)
            @info("relentr then geomean bridge")
            MOIT.contconictest(MOI.Bridges.Constraint.GeoMeantoRelEntr{T}(MOI.Bridges.Constraint.RelativeEntropy{T}(optimizer)), config, conic_exclude)
            @info("geomean then relentr bridge")
            MOIT.contconictest(MOI.Bridges.Constraint.RelativeEntropy{T}(MOI.Bridges.Constraint.GeoMeantoRelEntr{T}(optimizer)), config, conic_exclude)
        end
    end

    return
end
