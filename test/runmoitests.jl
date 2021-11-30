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
    model = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{T}()),
        Hypatia.Optimizer{T}(; default_tol_relax = 4),
    )
    MOI.set(model, MOI.Silent(), true)

    tol = 2 * sqrt(sqrt(eps(T)))
    config = MOI.Test.Config(
        T,
        atol = tol,
        rtol = tol,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ObjectiveBound,
        ],
    )

    excludes = String[
        # not implemented:
        "test_attribute_SolverVersion",
        # TODO fix:
        "test_model_copy_to_Unsupported",
    ]
    includes = String[]

    MOI.Test.runtests(model, config, include = includes, exclude = excludes)
end

end
;
