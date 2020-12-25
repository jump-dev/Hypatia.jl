#=
run MOI tests
=#

using Test
using Printf
import Hypatia
import Hypatia.Solvers
include(joinpath(@__DIR__, "moicones.jl"))
include(joinpath(@__DIR__, "moi.jl"))

@testset "MOI tests" begin

@testset "MOI wrapper cone tests" begin
    println("starting MOI wrapper cone tests")
    real_types = [
        Float64,
        Float32,
        BigFloat,
        ]
    for T in real_types
        println(T, " ...")
        test_moi_cones(T)
    end
end

default_options = (
    # verbose = true,
    verbose = false,
    default_tol_relax = 2,
    )

@testset "MOI.Test tests" begin
    println("\nstarting MOI.Test tests")
    options = [
        (Float64, false),
        (Float64, true),
        (Float32, true),
        (BigFloat, true),
        ]
    for (T, use_dense_model) in options
        test_info = "$T, $use_dense_model"
        @testset "$test_info" begin
            println(test_info, " ...")
            test_time = @elapsed test_moi(T, use_dense_model = use_dense_model; default_options...)
            @printf("%8.2e seconds\n", test_time)
        end
    end
end

end
;
