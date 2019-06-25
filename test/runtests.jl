#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Test
import Hypatia
import Hypatia.HypReal
import Hypatia.HypRealOrComplex
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models
const SO = HYP.Solvers
const MU = HYP.ModelUtilities

include(joinpath(@__DIR__, "interpolation.jl"))
include(joinpath(@__DIR__, "barriers.jl"))
include(joinpath(@__DIR__, "native.jl"))
include(joinpath(@__DIR__, "MathOptInterface.jl"))

examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "centralpolymat/JuMP.jl"))
include(joinpath(examples_dir, "envelope/native.jl"))
include(joinpath(examples_dir, "expdesign/native.jl"))
include(joinpath(examples_dir, "linearopt/native.jl"))
include(joinpath(examples_dir, "polymin/native.jl"))
include(joinpath(examples_dir, "contraction/JuMP.jl"))
include(joinpath(examples_dir, "densityest/JuMP.jl"))
include(joinpath(examples_dir, "densityest/native.jl"))
include(joinpath(examples_dir, "envelope/JuMP.jl"))
include(joinpath(examples_dir, "expdesign/JuMP.jl"))
include(joinpath(examples_dir, "lotkavolterra/JuMP.jl"))
include(joinpath(examples_dir, "muconvexity/JuMP.jl"))
include(joinpath(examples_dir, "polymin/JuMP.jl"))
include(joinpath(examples_dir, "polynorm/JuMP.jl"))
include(joinpath(examples_dir, "portfolio/native.jl"))
include(joinpath(examples_dir, "regionofattr/JuMP.jl"))
include(joinpath(examples_dir, "secondorderpoly/JuMP.jl"))
include(joinpath(examples_dir, "shapeconregr/JuMP.jl"))
include(joinpath(examples_dir, "semidefinitepoly/JuMP.jl"))

real_types = [
    Float64,
    Float32,
    BigFloat,
    ]

@info("starting Hypatia tests")
@testset "Hypatia tests" begin

@info("starting interpolation tests")
@testset "interpolation tests" begin
    fekete_sample()
    test_recover_lagrange_polys()
    test_recover_cheb_polys()
end

@info("starting barrier tests")
barrier_testfuns = [
    test_epinormeucl_barrier,
    test_epinorinf_barrier,
    test_epinormspectral_barrier,
    test_epipersquare_barrier,
    # TODO next 3 fail with BigFloat, see https://github.com/JuliaDiff/DiffResults.jl/pull/9#issuecomment-497853361
    test_epiperpower_barrier,
    test_epipersumexp_barrier,
    test_hypogeomean_barrier,
    test_hypoperlog_barrier,
    test_hypopersumlog_barrier,
    test_hypoperlogdet_barrier,
    test_semidefinite_barrier,
    test_wsospolyinterp_barrier,
    test_wsospolyinterpmat_barrier,
    test_wsospolyinterpsoc_barrier, # NOTE not updated for generic reals (too much work)
    ]
@testset "barrier functions tests: $t, $T" for t in barrier_testfuns, T in real_types
    t(T)
end

@info("starting native interface tests")
verbose = false
system_solvers = [
    SO.QRCholCombinedHSDSystemSolver,
    SO.SymIndefCombinedHSDSystemSolver,
    SO.NaiveElimCombinedHSDSystemSolver,
    SO.NaiveCombinedHSDSystemSolver,
    ]
testfuns_singular = [
    dimension1,
    consistent1,
    inconsistent1,
    inconsistent2,
    ]
@testset "preprocessing tests: $t, $s, $T" for t in testfuns_singular, s in system_solvers, T in real_types
    t(s{T}, MO.PreprocessedLinearModel{T}, verbose)
end
linear_models = [
    MO.PreprocessedLinearModel,
    MO.RawLinearModel,
    ]
testfuns_nonsingular = [
    orthant1,
    orthant2,
    orthant3,
    orthant4,
    epinorminf1,
    epinorminf2,
    epinorminf3,
    epinorminf4,
    epinorminf5,
    epinorminf6,
    epinormeucl1,
    epinormeucl2,
    epipersquare1,
    epipersquare2,
    epipersquare3,
    semidefinite1,
    semidefinite2,
    semidefinite3,
    semidefinitecomplex1,
    hypoperlog1,
    hypoperlog2,
    hypoperlog3,
    hypoperlog4,
    hypopersumlog1,
    hypopersumlog2,
    epiperpower1,
    epiperpower2,
    epiperpower3,
    hypogeomean1,
    hypogeomean2,
    hypogeomean3,
    hypogeomean4,
    epinormspectral1,
    hypoperlogdet1,
    hypoperlogdet2,
    hypoperlogdet3,
    epipersumexp1,
    epipersumexp2,
    ]
@testset "native tests: $t, $s, $m, $T" for t in testfuns_nonsingular, s in system_solvers, m in linear_models, T in real_types
    if s == SO.QRCholCombinedHSDSystemSolver && m == MO.RawLinearModel
        continue # QRChol linear system solver needs preprocessed model
    end
    t(s{T}, m{T}, verbose)
end

@info("starting MathOptInterface tests")
verbose = false
dense_options = [
    true,
    false,
    ]
system_solvers = [
    SO.NaiveElimCombinedHSDSystemSolver,
    SO.QRCholCombinedHSDSystemSolver,
    ]
linear_models = [
    MO.PreprocessedLinearModel, # some MOI tests require preprocessing to pass
    ]
@testset "MOI tests: $(d ? "dense" : "sparse"), $s, $m" for d in dense_options, s in system_solvers, m in linear_models
    test_moi(d, s{Float64}, m{Float64}, verbose)
end

@info("starting native examples tests")
native_options = (
    verbose = true,
    max_iters = 150,
    time_limit = 6e2, # 1 minute
    )
@testset "native examples" begin
    @testset "densityest" begin test_densityest(; native_options...,
        ) end
    @testset "envelope" begin test_envelope(; native_options...,
        ) end
    @testset "expdesign" begin test_expdesign(; native_options...,
        ) end
    @testset "linearopt" begin test_linearopt(; native_options...,
        ) end
    @testset "polymin" begin test_polymin(; native_options...,
        tol_rel_opt = 1e-9, tol_abs_opt = 1e-8, tol_feas = 1e-9,
        ) end
    @testset "portfolio" begin test_portfolio(; native_options...,
        ) end
end

@info("starting JuMP examples tests")
JuMP_options = (
    verbose = false,
    test_certificates = true,
    max_iters = 250,
    time_limit = 6e2, # 1 minute
    )
@testset "JuMP examples" begin
    @testset "centralpolymat" begin test_centralpolymatJuMP(; JuMP_options...,
        tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, tol_feas = 1e-7,
        ) end
    @testset "contraction" begin test_contractionJuMP(; JuMP_options...,
        tol_rel_opt = 1e-4, tol_abs_opt = 1e-4, tol_feas = 1e-4,
        ) end
    @testset "densityest" begin test_densityestJuMP(; JuMP_options...,
        tol_rel_opt = 1e-5, tol_abs_opt = 1e-5, tol_feas = 1e-6,
        ) end
    @testset "envelope" begin test_envelopeJuMP(; JuMP_options...,
        ) end
    @testset "expdesign" begin test_expdesignJuMP(; JuMP_options...,
        ) end
    @testset "lotkavolterra" begin test_lotkavolterraJuMP(; JuMP_options...,
        tol_rel_opt = 1e-5, tol_abs_opt = 1e-6, tol_feas = 1e-6,
        ) end
    @testset "muconvexity" begin test_muconvexityJuMP(; JuMP_options...,
        ) end
    @testset "polymin" begin test_polyminJuMP(; JuMP_options...,
        tol_rel_opt = 1e-9, tol_abs_opt = 1e-8, tol_feas = 1e-9,
        ) end
    @testset "polynorm" begin test_polynormJuMP(; JuMP_options...,
        ) end
    @testset "regionofattr" begin test_regionofattrJuMP(; JuMP_options...,
        tol_abs_opt = 1e-6, tol_rel_opt = 1e-6, tol_feas = 1e-6,
        ) end
    @testset "secondorderpoly" begin test_secondorderpolyJuMP(; JuMP_options...,
        ) end
    @testset "semidefinitepoly" begin test_semidefinitepolyJuMP(; JuMP_options...,
        tol_abs_opt = 1e-7, tol_rel_opt = 1e-7, tol_feas = 1e-7,
        ) end
    @testset "shapeconregr" begin test_shapeconregrJuMP(; JuMP_options...,
        tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, tol_feas = 1e-6,
        ) end
end

end
