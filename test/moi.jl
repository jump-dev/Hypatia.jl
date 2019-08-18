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
const MO = Hypatia.Models
const SO = Hypatia.Solvers

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
    "solve_zero_one_with_bounds_1",
    "solve_zero_one_with_bounds_2",
    "solve_zero_one_with_bounds_3",
    "solve_unbounded_model", # dual equalities are inconsistent, so detect dual infeasibility but currently no certificate or status
    ]

conic_exclude = String[
    # "lin",
    # "norminf",
    # "normone",
    # "soc",
    # "rsoc",
    # "exp",
    # "geomean",
    # "sdp",
    # "logdet",
    # "rootdet",
    # TODO currently some issue with square det transformation?
    "logdets",
    "rootdets",
    ]

function test_moi(
    use_dense::Bool,
    system_solver::Type{<:SO.CombinedHSDSystemSolver{T}},
    linear_model::Type{<:MO.LinearModel{T}},
    verbose::Bool,
    ) where {T <: Real}
    optimizer = MOIU.CachingOptimizer(
        MOIU.UniversalFallback(MOIU.Model{T}()),
        Hypatia.Optimizer{T}(
            use_dense = use_dense,
            test_certificates = true,
            verbose = verbose,
            system_solver = system_solver,
            linear_model = linear_model,
            max_iters = 200,
            time_limit = 2e1,
            tol_rel_opt = 2e-8,
            tol_abs_opt = 2e-8,
            tol_feas = 1e-8,
            tol_slow = 1e-7,
            )
        )

    @testset "unit tests" begin
        MOIT.unittest(optimizer, config, unit_exclude)
    end
    @testset "linear tests" begin
        MOIT.contlineartest(optimizer, config)
    end
    @testset "conic tests" begin
        MOIT.contconictest(MOIB.Constraint.Square{T}(MOIB.Constraint.RootDet{T}(optimizer)), config, conic_exclude)
    end

    return
end
