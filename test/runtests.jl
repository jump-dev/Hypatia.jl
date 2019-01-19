#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const LS = HYP.LinearSystems
const MU = HYP.ModelUtilities

using Random
using LinearAlgebra
using SparseArrays
using Test

examples_dir = joinpath(@__DIR__, "../examples")


# TODO make first part a native interface function eventually
# TODO maybe build a new high-level model struct; the current model struct is low-level
function solveandcheck(mdl, c, A, b, G, h, cone, linearsystem; atol=1e-4, rtol=1e-4)
    HYP.check_data(c, A, b, G, h, cone)
    if linearsystem == LS.QRSymm
        (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = HYP.preprocess_data(c, A, b, G, useQR=true)
        L = LS.QRSymm(c1, A1, b1, G1, h, cone, Q2, RiQ1)
    elseif linearsystem == LS.Naive
        (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = HYP.preprocess_data(c, A, b, G, useQR=false)
        L = LS.Naive(c1, A1, b1, G1, h, cone)
    else
        error("linear system cache type $linearsystem is not recognized")
    end
    HYP.load_data!(mdl, c1, A1, b1, G1, h, cone, L)
    HYP.solve!(mdl)

    # construct solution
    x = zeros(length(c))
    x[dukeep] = HYP.get_x(mdl)
    y = zeros(length(b))
    y[prkeep] = HYP.get_y(mdl)
    s = HYP.get_s(mdl)
    z = HYP.get_z(mdl)
    pobj = HYP.get_pobj(mdl)
    dobj = HYP.get_dobj(mdl)
    status = HYP.get_status(mdl)
    stime = HYP.get_solvetime(mdl)
    niters = HYP.get_niters(mdl)

    # check conic certificates are valid; conditions are described by CVXOPT at https://github.com/cvxopt/cvxopt/blob/master/src/python/coneprog.py
    # CO.loadpnt!(cone, s, z)
    if status == :Optimal
        # @test HYP.incone(cone)
        @test pobj ≈ dobj atol=atol rtol=rtol
        @test A*x ≈ b atol=atol rtol=rtol
        @test G*x + s ≈ h atol=atol rtol=rtol
        @test G'*z + A'*y ≈ -c atol=atol rtol=rtol
        @test dot(s, z) ≈ 0.0 atol=atol rtol=rtol
        @test dot(c, x) ≈ pobj atol=1e-8 rtol=1e-8
        @test dot(b, y) + dot(h, z) ≈ -dobj atol=1e-8 rtol=1e-8
    elseif status == :PrimalInfeasible
        # @test HYP.incone(cone)
        @test isnan(pobj)
        @test dobj > 0
        @test dot(b, y) + dot(h, z) ≈ -dobj atol=1e-8 rtol=1e-8
        @test G'*z ≈ -A'*y atol=atol rtol=rtol
    elseif status == :DualInfeasible
        # @test HYP.incone(cone)
        @test isnan(dobj)
        @test pobj < 0
        @test dot(c, x) ≈ pobj atol=1e-8 rtol=1e-8
        @test G*x ≈ -s atol=atol rtol=rtol
        @test A*x ≈ zeros(length(y)) atol=atol rtol=rtol
    elseif status == :IllPosed
        # @test HYP.incone(cone)
        # TODO primal vs dual ill-posed statuses and conditions
    end

    return (x=x, y=y, s=s, z=z, pobj=pobj, dobj=dobj, status=status, stime=stime, niters=niters)
end


@testset begin

@info("starting interpolation tests")
include(joinpath(@__DIR__, "interpolation.jl"))


include(joinpath(@__DIR__, "native.jl"))
@info("starting native interface tests")
verbose = false
linearsystems = [
    LS.QRSymm,
    LS.Naive,
    ]
testfuns = [
    dimension1,
    consistent1,
    inconsistent1,
    inconsistent2,
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
    hypoperlog1,
    hypoperlog2,
    hypoperlog3,
    hypoperlog4,
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
@testset "native tests: $t, $l" for t in testfuns, l in linearsystems
    t(verbose=verbose, linearsystem=l)
end


# load native interface examples
include(joinpath(examples_dir, "envelope/native.jl"))
include(joinpath(examples_dir, "linearopt/native.jl"))
include(joinpath(examples_dir, "namedpoly/native.jl"))

@info("starting default native examples tests")
testfuns = [
    run_envelope_primal_dense,
    run_envelope_dual_dense,
    run_envelope_primal_sparse,
    run_envelope_dual_sparse,
    run_linearopt,
    run_namedpoly,
    ]
@testset "default examples: $t" for t in testfuns
    t()
end

@info("starting additional native examples tests")
include(joinpath(@__DIR__, "examples.jl"))
verbose = false
linearsystems = [
    LS.QRSymm,
    # LS.Naive, # slow
    ]
testfuns = [
    envelope1,
    envelope2,
    envelope3,
    envelope4,
    linearopt1,
    linearopt2,
    namedpoly1,
    namedpoly2,
    namedpoly3,
    namedpoly4,
    namedpoly5,
    namedpoly6,
    namedpoly7,
    namedpoly8,
    namedpoly9,
    namedpoly10,
    namedpoly11,
    ]
@testset "native examples: $t, $l" for t in testfuns, l in linearsystems
    t(verbose=verbose, linearsystem=l)
end


# load JuMP examples
# include(joinpath(examples_dir, "envelope/jump.jl"))
# include(joinpath(examples_dir, "expdesign/jump.jl"))
# include(joinpath(examples_dir, "namedpoly/jump.jl"))
# include(joinpath(examples_dir, "shapeconregr/jump.jl"))
# include(joinpath(examples_dir, "densityest/jump.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmatrix.jl"))
# include(joinpath(examples_dir, "wsosmatrix/muconvexity.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmat1.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmat2.jl"))
# include(joinpath(examples_dir, "wsosmatrix/sosmat3.jl"))

# @info("starting default JuMP examples tests")
# testfuns = [
#     run_JuMP_expdesign,
#     run_JuMP_namedpoly_PSD, # final objective doesn't match
#     run_JuMP_namedpoly_WSOS_primal,
#     run_JuMP_namedpoly_WSOS_dual,
#     run_JuMP_envelope_boxinterp,
#     run_JuMP_envelope_sampleinterp_box,
#     run_JuMP_envelope_sampleinterp_ball,
#     run_JuMP_shapeconregr_PSD,
#     run_JuMP_shapeconregr_WSOS,
#     run_JuMP_densityest,
#     run_JuMP_sosmat4_matrix_rand,
#     run_JuMP_sosmat4_matrix_a,
#     run_JuMP_sosmat4_poly_a,
#     run_JuMP_sosmat4_poly_b,
#     run_JuMP_muconvexity_rand,
#     run_JuMP_muconvexity_a,
#     run_JuMP_muconvexity_b,
#     run_JuMP_muconvexity_c,
#     run_JuMP_muconvexity_d,
#     run_JuMP_sosmat1,
#     run_JuMP_sosmat2_scalar,
#     run_JuMP_sosmat2_matrix,
#     run_JuMP_sosmat2_matrix_dual,
#     run_JuMP_sosmat3, # slow
#     ]
# @testset "default examples: $t" for t in testfuns
#     t()
# end
#
# @info("starting additional JuMP examples tests")
# testfuns = [
#     namedpoly1_JuMP,
#     namedpoly2_JuMP,
#     namedpoly3_JuMP,
#     namedpoly4_JuMP, # numerically unstable
#     namedpoly5_JuMP,
#     namedpoly6_JuMP,
#     namedpoly7_JuMP,
#     namedpoly8_JuMP,
#     namedpoly9_JuMP,
#     namedpoly10_JuMP,
#     shapeconregr1_JuMP,
#     shapeconregr2_JuMP,
#     shapeconregr3_JuMP,
#     shapeconregr4_JuMP,
#     shapeconregr5_JuMP,
#     shapeconregr6_JuMP,
#     shapeconregr7_JuMP, # numerically unstable
#     shapeconregr8_JuMP,
#     shapeconregr9_JuMP, # numerically unstable
#     shapeconregr10_JuMP, # numerically unstable
#     shapeconregr11_JuMP, # numerically unstable
#     shapeconregr12_JuMP, # numerically unstable
#     shapeconregr13_JuMP, # numerically unstable
#     # shapeconregr14_JuMP, # throws out-of-memory error
#     # shapeconregr15_JuMP, # throws out-of-memory error
#     ]
# @testset "JuMP examples: $t" for t in testfuns
#     t()
# end
#

@info("starting MathOptInterface tests")
include(joinpath(@__DIR__, "MOI_wrapper.jl"))
verbose = false
linearsystems = [
    LS.QRSymm,
    LS.Naive,
    ]
@testset "MOI tests: $l, $(d ? "dense" : "sparse")" for l in linearsystems, d in [false, true]
    testmoi(verbose=verbose, linearsystem=l, usedense=d)
end

end
