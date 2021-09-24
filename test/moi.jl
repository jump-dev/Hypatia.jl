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
        Hypatia.Optimizer{T}(; solver_options...),
    )

    tol = 2sqrt(sqrt(eps(T)))
    config = MOIT.Config(
        T,
        atol = tol,
        rtol = tol,
        exclude = Any[MOI.VariableBasisStatus, MOI.ConstraintBasisStatus]
    )

    @testset "linear tests" begin
        @info optimizer
        MOIT.runtests(
            optimizer, config, include=["linear"],
            exclude=["test_linear_INFEASIBLE_2", "Indicator", "Integer"],
        )
    end

    if T == Float64
        # test other real types, waiting for https://github.com/jump-dev/MathOptInterface.jl/issues/841
        @testset "conic tests" begin
            config_conic = MOIT.Config(
                T,
                atol = 2tol,
                rtol = 2tol,
                exclude = Any[MOI.VariableBasisStatus, MOI.ConstraintBasisStatus]
            )
            MOIT.runtests(
                optimizer, config_conic, include = ["conic"],
                exclude=[
                    "linear",
                    "test_conic_RelativeEntropyCone",
                    "test_conic_SecondOrderCone_negative_post_bound_ii", # MathOptInterface.OTHER_ERROR instead of MathOptInterface.DUAL_INFEASIBLE
                    "test_conic_SecondOrderCone_no_initial_bound",
                ],
            )
        end
    end
    return
end
