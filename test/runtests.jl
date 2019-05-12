#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models
const SO = HYP.Solvers
const MU = HYP.ModelUtilities

import Random
using LinearAlgebra
using SparseArrays
using Test


include(joinpath(@__DIR__, "interpolation.jl"))
include(joinpath(@__DIR__, "barriers.jl"))

examples_dir = joinpath(@__DIR__, "../examples")

include(joinpath(examples_dir, "envelope/native.jl"))
include(joinpath(examples_dir, "linearopt/native.jl"))
include(joinpath(examples_dir, "polymin/real.jl"))
include(joinpath(examples_dir, "polymin/complex.jl"))
include(joinpath(examples_dir, "polymin/instances.jl"))

include(joinpath(@__DIR__, "native.jl"))

include(joinpath(@__DIR__, "MathOptInterface.jl"))

include(joinpath(examples_dir, "envelope/jump.jl"))
include(joinpath(examples_dir, "expdesign/jump.jl"))
include(joinpath(examples_dir, "polymin/jump.jl"))
include(joinpath(examples_dir, "shapeconregr/jump.jl"))
include(joinpath(examples_dir, "densityest/jump.jl"))
include(joinpath(examples_dir, "wsosmatrix/sosmatrix.jl"))
include(joinpath(examples_dir, "wsosmatrix/muconvexity.jl"))
include(joinpath(examples_dir, "wsosmatrix/sosmat1.jl"))
include(joinpath(examples_dir, "wsosmatrix/sosmat2.jl"))
include(joinpath(examples_dir, "wsosmatrix/sosmat3.jl"))
include(joinpath(examples_dir, "regionofattraction/univariate.jl"))
include(joinpath(examples_dir, "contractionanalysis/jump.jl"))

include(joinpath(@__DIR__, "JuMP.jl"))


@testset "Hypatia tests" begin

# @info("starting interpolation tests")
# @testset "interpolation tests" begin
#     fekete_sample()
#     test_recover_lagrange_polys()
#     test_recover_cheb_polys()
# end

# @info("starting barrier tests")
# barrier_testfuns = [
#     test_epinormeucl_barrier,
#     test_epinorinf_barrier,
#     test_epinormspectral_barrier,
#     test_epiperpower_barrier,
#     test_epipersquare_barrier,
#     test_epipersumexp_barrier,
#     test_hypogeomean_barrier,
#     test_hypoperlog_barrier,
#     test_hypoperlogdet_barrier,
#     test_semidefinite_barrier,
#     test_wsospolyinterp_2_barrier,
#     test_wsospolyinterp_barrier,
#     test_wsospolyinterpmat_barrier,
#     test_wsospolyinterpsoc_barrier,
# ]
# @testset "barrier functions tests: $t" for t in barrier_testfuns
#     t()
# end

# @info("starting native interface tests")
# verbose = true
# system_solvers = [
#     SO.NaiveCombinedHSDSystemSolver,
#     SO.QRCholCombinedHSDSystemSolver,
#     ]
# testfuns_singular = [
#     dimension1,
#     consistent1,
#     inconsistent1,
#     inconsistent2,
#     ]
# @testset "preprocessing tests: $t, $s" for t in testfuns_singular, s in system_solvers
#     t(s, MO.PreprocessedLinearModel, verbose)
# end
# linear_models = [
#     MO.RawLinearModel,
#     MO.PreprocessedLinearModel,
#     ]
# testfuns_nonsingular = [
#     orthant1,
#     orthant2,
#     orthant3,
#     orthant4,
#     epinorminf1,
#     epinorminf2,
#     epinorminf3,
#     epinorminf4,
#     epinorminf5,
#     epinorminf6,
#     epinormeucl1,
#     epinormeucl2,
#     epipersquare1,
#     epipersquare2,
#     epipersquare3,
#     semidefinite1,
#     semidefinite2,
#     semidefinite3,
#     hypoperlog1,
#     hypoperlog2,
#     hypoperlog3,
#     hypoperlog4,
#     epiperpower1,
#     epiperpower2,
#     epiperpower3,
#     hypogeomean1,
#     hypogeomean2,
#     hypogeomean3,
#     hypogeomean4,
#     epinormspectral1,
#     hypoperlogdet1,
#     hypoperlogdet2,
#     hypoperlogdet3,
#     epipersumexp1,
#     epipersumexp2,
#     ]
# @testset "native tests: $t, $s, $m" for t in testfuns_nonsingular, s in system_solvers, m in linear_models
#     if s == SO.QRCholCombinedHSDSystemSolver && m == MO.RawLinearModel
#         continue # QRChol linear system solver needs preprocessed model
#     end
#     t(s, m, verbose)
# end

# @info("starting default native examples tests")
# testfuns = [
#     run_envelope_primal_dense,
#     run_envelope_dual_dense,
#     run_envelope_primal_sparse,
#     run_envelope_dual_sparse,
#     run_linearopt,
#     run_polymin,
#     run_complexpolymin_primal,
#     run_complexpolymin_dual,
#     ]
# @testset "default examples: $t" for t in testfuns
#     t()
# end

# @info("starting additional native examples tests")
# verbose = true
# system_solvers = [
#     # SO.NaiveCombinedHSDSystemSolver,
#     SO.QRCholCombinedHSDSystemSolver,
#     ]
# linear_models = [
#     # MO.RawLinearModel,
#     MO.PreprocessedLinearModel,
#     ]
# testfuns = [
#     # TODO test primal and dual formulations of envelope
#     envelope1,
#     envelope2,
#     envelope3,
#     envelope4,
#     linearopt1,
#     linearopt2,
#     polymin1,
#     polymin2,
#     polymin3,
#     polymin4,
#     polymin5,
#     polymin6,
#     polymin7,
#     polymin8,
#     polymin9,
#     polymin10,
#     polymin11,
#     test_complexpolymin1,
#     test_complexpolymin2,
#     test_complexpolymin3,
#     test_complexpolymin4,
#     test_complexpolymin5,
#     test_complexpolymin6,
#     test_complexpolymin7,
#     ]
# @testset "native examples: $t, $s, $m" for t in testfuns, s in system_solvers, m in linear_models
#     if s == SO.QRCholCombinedHSDSystemSolver && m == MO.RawLinearModel
#         continue # QRChol linear system solver needs preprocessed model
#     end
#     t(s, m, verbose)
# end

# @info("starting MathOptInterface tests")
# verbose = false
# system_solvers = [
#     SO.NaiveCombinedHSDSystemSolver,
#     SO.QRCholCombinedHSDSystemSolver,
#     ]
# linear_models = [
#     MO.PreprocessedLinearModel, # MOI tests require preprocessing
#     ]
# @testset "MOI tests: $(d ? "dense" : "sparse"), $s, $m" for d in (false, true), s in system_solvers, m in linear_models
#     test_moi(d, s, m, verbose)
# end

# @info("starting default JuMP examples tests")
# testfuns = [
#     run_JuMP_envelope_boxinterp,
#     run_JuMP_envelope_sampleinterp_box,
#     run_JuMP_envelope_sampleinterp_ball,
#     run_JuMP_expdesign,
#     # run_JuMP_polymin_PSD, # too slow TODO check: final objective doesn't match
#     run_JuMP_polymin_WSOS_primal,
#     run_JuMP_polymin_WSOS_dual,
#     run_JuMP_shapeconregr_PSD,
#     run_JuMP_shapeconregr_WSOS,
#     run_JuMP_shapeconregr_WSOS_PolyJuMP,
#     run_JuMP_densityest,
#     run_JuMP_sosmatrix_rand,
#     run_JuMP_sosmatrix_a,
#     run_JuMP_sosmatrix_poly_a,
#     run_JuMP_sosmatrix_poly_b,
#     run_JuMP_muconvexity_rand,
#     run_JuMP_muconvexity_a,
#     run_JuMP_muconvexity_b,
#     run_JuMP_muconvexity_c,
#     run_JuMP_muconvexity_d,
#     run_JuMP_sosmat1,
#     run_JuMP_sosmat2_scalar,
#     run_JuMP_sosmat2_matrix,
#     run_JuMP_sosmat2_matrix_dual,
#     run_JuMP_sosmat3_primal, # numerically unstable
#     run_JuMP_sosmat3_dual, # numerically unstable
#    # run_JuMP_univariate_roa_WSOS,
#    # run_JuMP_univariate_roa_PSD,
#    # run_JuMP_contraction_PSD,
#    # run_JuMP_contraction_WSOS,
#     ]
# @testset "default examples: $t" for t in testfuns
#     t()
# end

@info("starting additional JuMP examples tests")
testfuns = [
    test_polymin_JuMP_many,
    test_shapeconregr_JuMP_many,
    ]
@testset "JuMP examples: $t" for t in testfuns
    t()
end

end
