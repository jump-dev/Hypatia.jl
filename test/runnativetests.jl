#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "native.jl"))

const MO = Hypatia.Models
const SO = Hypatia.Solvers

@info("starting native tests")

real_types = [
    Float64,
    Float32,
    BigFloat,
    ]

system_solvers = [
    SO.QRCholHSDSystemSolver,
    SO.SymIndefHSDSystemSolver,
    SO.NaiveElimHSDSystemSolver,
    SO.NaiveHSDSystemSolver,
    ]

testfuns_preproc = [
    dimension1,
    consistent1,
    inconsistent1,
    inconsistent2,
    ]

linear_models = [
    MO.PreprocessedLinearModel,
    MO.RawLinearModel,
    ]

use_infty_nbhd = [
    true,
    false,
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

@testset "preprocessing tests: $t, $s, $T" for t in testfuns_preproc, s in system_solvers, T in real_types
    t(T, solver = SO.HSDSolver{T}(verbose = true, system_solver = s{T}()))
end

@testset "native tests: $t, $s, $m, $n, $T" for t in testfuns_raw, s in system_solvers, m in linear_models, n in use_infty_nbhd, T in real_types
    T == BigFloat && t == epinormspectral1 && continue # Cannot get svdvals with BigFloat
    s == SO.QRCholHSDSystemSolver && m == MO.RawLinearModel && continue # QRChol linear system solver needs preprocessed model
    t(T, linear_model = m, solver = SO.HSDSolver{T}(verbose = false, use_infty_nbhd = n, system_solver = s{T}()))
end

@testset "native tests (iterative linear system solves): $t, $T" for t in testfuns_raw, T in real_types
    T == BigFloat && continue # IterativeSolvers does not work with BigFloat
    t(T, linear_model = MO.RawLinearModel, linear_model_options = (use_iterative = true,),
        solver = SO.HSDSolver{T}(verbose = true, system_solver = SO.NaiveHSDSystemSolver{T}(use_iterative = true)))
end
