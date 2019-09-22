#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "native.jl"))

const SO = Hypatia.Solvers

real_types = [
    Float64,
    Float32,
    BigFloat,
    ]

system_solvers = [
    SO.QRCholDenseSystemSolver,
    SO.SymIndefDenseSystemSolver,
    SO.SymIndefSparseSystemSolver,
    SO.NaiveElimDenseSystemSolver,
    SO.NaiveDenseSystemSolver,
    SO.NaiveSparseSystemSolver,
    SO.NaiveIndirectSystemSolver,
    ]

use_infty_nbhd = [
    true,
    # false,
    ]

preprocess = [
    true,
    # false
    ]

testfuns_preproc = [
    dimension1,
    consistent1,
    inconsistent1,
    inconsistent2,
    ]

testfuns_raw = [
    orthant1,
    orthant2,
    orthant3,
    orthant4,
    epinorminf1,
    epinorminf2,
    epinorminf3,
    epinorminf4,
    epinorminf5,
    epinormeucl1,
    epinormeucl2,
    epipersquare1,
    epipersquare2,
    epipersquare3,
    hypoperlog1,
    hypoperlog2,
    hypoperlog3,
    hypoperlog4,
    hypoperlog5,
    hypoperlog6,
    epiperexp1,
    epiperexp2,
    power1,
    power2,
    power3,
    power4,
    hypogeomean1,
    hypogeomean2,
    hypogeomean3,
    epinormspectral1,
    possemideftri1,
    possemideftri2,
    possemideftricomplex1,
    hypoperlogdettri1,
    hypoperlogdettri2,
    hypoperlogdettri3,
    primalinfeas1,
    primalinfeas2,
    primalinfeas3,
    dualinfeas1,
    dualinfeas2,
    dualinfeas3,
    ]

@info("starting native tests")
@testset "native tests" begin
    @info("starting preprocessing tests")
    @testset "preprocessing tests: $t, $T" for t in testfuns_preproc, T in real_types
        t(T, solver = SO.Solver{T}(verbose = true, system_solver = SO.QRCholDenseSystemSolver{T}()))
    end

    # TODO remove - both options should work
    # @test_throws Exception SO.SymIndefSparseSystemSolver(use_inv_hess = false)

    # TODO test options to system solvers

    @info("starting miscellaneous tests")
    @testset "miscellaneous tests: $t, $s, $n, $p, $T" for t in testfuns_raw, s in system_solvers, n in use_infty_nbhd, p in preprocess, T in real_types
        !p && s == SO.QRCholSystemSolver && continue # must use preprocessing if using QRCholSystemSolver
        T == BigFloat && t == epinormspectral1 && continue # cannot get svdvals with BigFloat
        T == BigFloat && s == SO.NaiveIndirectSystemSolver && continue # cannot use indirect methods with BigFloat
        T != Float64 && s in (SO.SymIndefSparseSystemSolver, SO.NaiveSparseSystemSolver) && continue # sparse system solvers only work with Float64
        solver = SO.Solver{T}(verbose = false, preprocess = p, use_infty_nbhd = n, system_solver = s{T}())
        t(T, solver = solver)
    end
end
