#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
run MOI tests
=#

using Test
import MathOptInterface as MOI
import Hypatia
include(joinpath(@__DIR__, "moicones.jl"))

@testset "MathOptInterface wrapper tests" begin
    @testset "MOI cone tests" begin
        println("starting MOI wrapper cone tests")
        real_types = [Float64, BigFloat]
        for T in real_types
            @testset "$T" begin
                println(T, " ...")
                test_moi_cones(T)
            end
        end
    end

    # real types, tolerances, and tests to include for MOI.Test tests
    test_T = [
        (Float64, 2 * sqrt(sqrt(eps())), 4, String[], String[]),
        (
            BigFloat,
            2 * eps(BigFloat)^0.15,
            1,
            String["test_linear", "test_conic", "test_modification"],
            String["test_linear_INFEASIBLE_2",],
        ),
    ]

    @testset "MOI.Test tests: $T" for (T, tol_test, tol_relax, includes, excludes) in test_T
        println("\nstarting MOI.Test tests: $T")
        model = MOI.Bridges.full_bridge_optimizer(
            MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{T}()),
                Hypatia.Optimizer{T}(; default_tol_relax = tol_relax),
            ),
            T,
        )
        MOI.set(model, MOI.Silent(), true)
        MOI.Test.runtests(
            model,
            MOI.Test.Config(
                T,
                atol = tol_test,
                rtol = tol_test,
                exclude = Any[
                    MOI.ConstraintBasisStatus,
                    MOI.VariableBasisStatus,
                    MOI.ObjectiveBound,
                    MOI.SolverVersion,
                ],
            ),
            include = includes,
            exclude = vcat(
                excludes,
                String[
                    # TODO(odow): unexpected failure, probably in the bridge layer
                    "test_model_UpperBoundAlreadySet",
                    "test_model_LowerBoundAlreadySet",
                ],
            ),
        )
    end
end;
