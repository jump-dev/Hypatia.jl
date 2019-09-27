#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "moi.jl"))

dense_system_solvers = [
    SO.NaiveDenseSystemSolver,
    SO.NaiveIndirectSystemSolver,
    SO.NaiveElimDenseSystemSolver,
    SO.SymIndefDenseSystemSolver,
    SO.QRCholDenseSystemSolver,
    ]

sparse_system_solvers = [
    SO.NaiveSparseSystemSolver,
    SO.NaiveIndirectSystemSolver,
    SO.NaiveElimSparseSystemSolver,
    SO.SymIndefSparseSystemSolver,
    ]

real_types = [
    Float64,
    # BigFloat, # TODO test this when MOI allows
    ]

verbose = false

@info("starting MOI tests")
@testset "MOI tests" begin
    @testset "MOI tests with dense A/G: $s, $T" for s in dense_system_solvers, T in real_types
        test_moi(T, true, verbose = verbose, system_solver = s{T}())
    end
    @testset "MOI tests with sparse A/G: $s, $T" for s in sparse_system_solvers, T in real_types
        test_moi(T, false, verbose = verbose, system_solver = s{T}())
    end
end
