#=
run MOI tests
=#

using Test
import Hypatia
import Hypatia.Solvers

# MOI wrapper Hypatia cone tests
include(joinpath(@__DIR__, "moicones.jl"))

real_types = [
    Float64,
    Float32,
    BigFloat,
    ]

@testset "MOI wrapper Hypatia cone tests" begin
    @testset "MOI wrapper Hypatia cone tests: $T" for T in real_types
        test_moi_cones(T)
    end
end

# MOI.Test linear and conic tests
include(joinpath(@__DIR__, "moi.jl"))

system_solvers = [
    # Solvers.NaiveDenseSystemSolver,
    # Solvers.NaiveSparseSystemSolver,
    # Solvers.NaiveElimDenseSystemSolver,
    # Solvers.NaiveElimSparseSystemSolver,
    # Solvers.SymIndefDenseSystemSolver,
    Solvers.SymIndefSparseSystemSolver,
    Solvers.QRCholDenseSystemSolver,
    ]

real_types = [
    Float64,
    # TODO test when wrapper allows
    # Float32,
    # BigFloat,
    ]

@testset "MOI.Test tests" begin
@testset "MOI.Test tests: $s, $T" for s in system_solvers, T in real_types, use_dense_model in (false, true)
    test_moi(T, use_dense_model, verbose = false, system_solver = s{T}())
end
end
