#=
run MOI tests
=#

using Test
import MathOptInterface
const MOI = MathOptInterface
import Hypatia
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
    model = MOI.Bridges.full_bridge_optimizer(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{T}()),
            Hypatia.Optimizer{T}(; default_tol_relax = 4),
        ),
        T,
    )
    MOI.set(model, MOI.Silent(), true)
    MOI.Test.runtests(
        model,
        MOI.Test.Config(
            T,
            atol = 2 * sqrt(sqrt(eps(T))),
            rtol = 2 * sqrt(sqrt(eps(T))),
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ObjectiveBound,
                MOI.SolverVersion,
            ],
        ),
        exclude = String[
            # TODO(odow): unexpected failure. But this is probably in the bridge
            # layer, not Hypatia.
            "test_model_UpperBoundAlreadySet",
            "test_model_LowerBoundAlreadySet",
        ],
    )
end

end
;
