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

include(joinpath(@__DIR__, "native.jl"))

include(joinpath(@__DIR__, "MathOptInterface.jl"))

include(joinpath(examples_dir, "envelope/jump.jl"))
include(joinpath(examples_dir, "expdesign/jump.jl"))
include(joinpath(examples_dir, "lotkavolterra/jump.jl"))
include(joinpath(examples_dir, "polymin/jump.jl"))
include(joinpath(examples_dir, "shapeconregr/jump.jl"))
include(joinpath(examples_dir, "densityest/jump.jl"))
include(joinpath(examples_dir, "wsosmatrix/sosmatrix.jl"))
include(joinpath(examples_dir, "wsosmatrix/muconvexity.jl"))
include(joinpath(examples_dir, "wsosmatrix/sosmat1.jl"))
include(joinpath(examples_dir, "wsosmatrix/sosmat2.jl"))
include(joinpath(examples_dir, "wsosmatrix/sosmat3.jl"))
include(joinpath(examples_dir, "regionofattraction/univariate.jl"))
include(joinpath(examples_dir, "contraction/jump.jl"))


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
#     test_envelopes,
#     test_linearopts,
#     test_polymins,
#     test_complexpolymins,
#     ]
# @testset "default examples: $t" for t in testfuns
#     t()
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
#     # run_JuMP_polymin_PSD, # too slow TODO check: final objective doesn't match
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
#     ]
# @testset "default examples: $t" for t in testfuns
#     t()
# end
#
@info("starting additional JuMP examples tests")
@testset "JuMP examples" begin
    test_contraction_JuMP(verbose = true, tol_rel_opt = 1e-4, tol_abs_opt = 1e-4, tol_feas = 1e-4)
    test_densityest_JuMP(verbose = true)
    test_envelope_JuMP(verbose = true)
    test_expdesign_JuMP(verbose = true)
    # test_lotkavolterra_JuMP(verbose = true, max_iters = 1000, time_limit = 3.6e4, tol_rel_opt = 1e-5, tol_abs_opt = 1e-6, tol_feas = 1e-6)
    test_polymin_JuMP(verbose = true, tol_rel_opt = 1e-8, tol_abs_opt = 1e-8, tol_feas = 1e-8)
    test_roauniv_JuMP(verbose = true, tol_feas = 1e-5)
    # # test_shapeconregr_JuMP(verbose = true)
end

end
