#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "native.jl"))
using TimerOutputs

const SO = Hypatia.Solvers

real_types = [
    Float64,
    Float32,
    BigFloat,
    ]

system_solvers = [
    SO.QRCholDenseSystemSolver,
    SO.SymIndefDenseSystemSolver,
    SO.NaiveElimDenseSystemSolver,
    SO.NaiveDenseSystemSolver,
    SO.NaiveSparseSystemSolver,
    SO.SymIndefSparseSystemSolver,
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

# @info("starting preprocessing tests")
# @testset "preprocessing tests: $t, $s, $T" for t in testfuns_preproc, s in system_solvers, T in real_types
#     t(T, solver = SO.Solver{T}(verbose = true, system_solver = s{T}()))
# end

tol = 1e-8

@info("starting miscellaneous tests")
@testset "miscellaneous tests: $t, $s, $n, $p, $T" for t in testfuns_raw, s in system_solvers, n in use_infty_nbhd, p in preprocess, T in real_types
    T == BigFloat && t == epinormspectral1 && continue # Cannot get svdvals with BigFloat
    T == BigFloat && isa(s, SO.NaiveIndirectSystemSolver) && continue
    !p && s == SO.QRCholSystemSolver && continue # Must use preprocessing if using QRCholSystemSolver
    solver = SO.Solver{T}(verbose = false, preprocess = p, use_infty_nbhd = n, system_solver = s{T}())
    t(T, solver = solver)
end

# @info("starting iterative system solver tests")
# @testset "iterative system solver tests: $t, $T" for t in testfuns_raw, T in real_types
#     T == BigFloat && continue # IterativeSolvers does not work with BigFloat
#     solver = SO.Solver{T}(verbose = true, init_use_indirect = true, preprocess = false,
#         system_solver = SO.NaiveSystemSolver{T}(use_indirect = true))
#     t(T, solver = solver)
# end
