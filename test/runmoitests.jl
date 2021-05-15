#=
run MOI tests
=#

using Test
using Printf
import Hypatia
import Hypatia.Solvers
include(joinpath(@__DIR__, "moicones.jl"))
include(joinpath(@__DIR__, "moi.jl"))

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
    options = [
        (Float64, Solvers.CombinedStepper, false),
        # (Float64, Solvers.CombinedStepper, true),
        # (Float32, Solvers.CombinedStepper, true),
        (BigFloat, Solvers.PredOrCentStepper, true),
        ]
    for (T, stepper, use_dense_model) in options
        default_options = (
            # verbose = true,
            verbose = false,
            default_tol_relax = 4,
            stepper = stepper{T}(),
            )
        test_info = "$T, $use_dense_model"
        @testset "$test_info" begin
            println(test_info, " ...")
            test_time = @elapsed test_moi(T,
                use_dense_model = use_dense_model; default_options...)
            @printf("%8.2e seconds\n", test_time)
        end
    end
end

end
;
