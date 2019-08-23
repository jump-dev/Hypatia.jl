#=
Copyright 2019, Chris Coey and contributors
=#

include(joinpath(@__DIR__, "native.jl"))

const MO = Hypatia.Models
const SO = Hypatia.Solvers

real_types = [
    Float64,
    Float32,
    BigFloat,
    ]

system_solvers = [
    SO.QRCholCombinedHSDSystemSolver,
    SO.SymIndefCombinedHSDSystemSolver,
    SO.NaiveElimCombinedHSDSystemSolver,
    SO.NaiveCombinedHSDSystemSolver,
    ]

testfuns_preproc = [
    dimension1,
    consistent1,
    inconsistent1,
    inconsistent2,
    ]

@info("starting preprocessing tests")
@testset "preprocessing tests: $t, $s, $T" for t in testfuns_preproc, s in system_solvers, T in real_types
    test_options = (
        linear_model = MO.PreprocessedLinearModel,
        system_solver = s,
        solver_options = (verbose = true,),
        )
    t(T, test_options)
end

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
    ]

@info("starting native tests")
@testset "native tests: $t, $s, $m, $n, $T" for t in testfuns_raw, s in system_solvers, m in linear_models, n in use_infty_nbhd, T in real_types
    if T == BigFloat && t == epinormspectral1
        continue # cannot get svdvals with BigFloat
    end
    if s == SO.QRCholCombinedHSDSystemSolver && m == MO.RawLinearModel
        continue # QRChol linear system solver needs preprocessed model
    end
    test_options = (
        linear_model = m,
        system_solver = s,
        linear_model_options = NamedTuple(),
        system_solver_options = NamedTuple(),
        stepper_options = (use_infty_nbhd = n,),
        solver_options = (verbose = false,),
        )
    t(T, test_options)
end

@info("starting native tests (iterative linear system methods)")
@testset "native tests (iterative linear system methods): $t, $T" for t in testfuns_raw, T in real_types
    if T == BigFloat
        continue # IterativeSolvers does not work with BigFloat
    end
    test_options = (
        linear_model = MO.RawLinearModel,
        system_solver = SO.NaiveCombinedHSDSystemSolver,
        linear_model_options = (use_iterative = true,),
        system_solver_options = (use_iterative = true,),
        solver_options = (verbose = false,),
        )
    t(T, test_options)
end
