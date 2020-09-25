#=
run MOI tests
=#

using Test
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

@testset "MOI.Test tests" begin
    println("\nstarting MOI.Test tests")
    system_solvers = [
        Solvers.SymIndefSparseSystemSolver,
        # Solvers.QRCholDenseSystemSolver,
        ]
    real_types = [
        Float64,
        # TODO test when wrapper allows
        # Float32,
        # BigFloat,
        ]
    dense_flags = [
        false,
        # true,
        ]
    for s in system_solvers, T in real_types, d in dense_flags
        test_info = "$s, $T, $d"
        @testset "$test_info" begin
            println(test_info, " ...")
            test_time = @elapsed test_moi(T, use_dense_model = d, verbose = false, system_solver = s{T}())
            @printf("%8.2e seconds\n", test_time)
        end
    end
end

end
;
