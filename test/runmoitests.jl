#=
run MOI tests
=#

using Test
using Printf
import MathOptInterface
const MOI = MathOptInterface
import Hypatia
import Hypatia.Solvers
include(joinpath(@__DIR__, "moicones.jl"))

@testset "MathOptInterface wrapper tests" begin

@testset "MOI cone tests" begin
    println("starting MOI wrapper cone tests")
    real_types = [
        Float64,
        # Float32,
        BigFloat,
        ]
    for T in real_types
        @testset "$T" begin
            println(T, " ...")
            test_moi_cones(T)
        end
    end
end

@testset "MOI.Test tests" begin
    println("\nstarting MOI.Test tests")
    # TODO test other real types
    T = Float64
    optimizer = Hypatia.Optimizer{T}(; default_tol_relax = 4)
    MOI.set(optimizer, MOI.Silent(), true)
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{T}()),
            optimizer,
        ),
        T,
    )

    tol = 2 * sqrt(sqrt(eps(T)))
    config = MOI.Test.Config(
        T,
        atol = tol,
        rtol = tol,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
        ],
    )

    excludes = String[
        # not implemented
        "test_attribute_SolverVersion",
        # TODO fix
        "test_linear_INFEASIBLE_2", # slow progress
        "test_solve_result_index",
        "test_model_copy_to_UnsupportedAttribute",
        # MathOptInterface.jl issue #1431
        "test_model_LowerBoundAlreadySet",
        "test_model_UpperBoundAlreadySet",
    ]
    includes = String[]

    MOI.Test.runtests(model, config, include = includes, exclude = excludes)
end

end
;
