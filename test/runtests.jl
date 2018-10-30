#=
Copyright 2018, Chris Coey and contributors
# TODO add a progress meter to silent tests?
# TODO don't print "Hypatia." before linsyscache types in testset printing
=#

using Hypatia
using Test
using Random
using LinearAlgebra
using SparseArrays


# TODO make first part a native interface function eventually
# TODO maybe build a new high-level model struct; the current model struct is low-level
function solveandcheck(
    mdl::Hypatia.Model,
    c,
    A,
    b,
    G,
    h,
    cone,
    lscachetype;
    atol = 1e-4,
    rtol = 1e-4,
    )
    # check, preprocess, load, and solve
    Hypatia.check_data(c, A, b, G, h, cone)
    if lscachetype == Hypatia.QRSymmCache
        (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)
        L = Hypatia.QRSymmCache(c1, A1, b1, G1, h, cone, Q2, RiQ1)
    elseif lscachetype == Hypatia.NaiveCache
        (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=false)
        L = Hypatia.NaiveCache(c1, A1, b1, G1, h, cone)
    else
        error("linear system cache type $lscachetype is not recognized")
    end
    Hypatia.load_data!(mdl, c1, A1, b1, G1, h, cone, L)
    Hypatia.solve!(mdl)

    # construct solution
    x = zeros(length(c))
    x[dukeep] = Hypatia.get_x(mdl)
    y = zeros(length(b))
    y[prkeep] = Hypatia.get_y(mdl)
    s = Hypatia.get_s(mdl)
    z = Hypatia.get_z(mdl)
    pobj = Hypatia.get_pobj(mdl)
    dobj = Hypatia.get_dobj(mdl)

    status = Hypatia.get_status(mdl)

    # check conic certificates are valid; conditions are described by CVXOPT at https://github.com/cvxopt/cvxopt/blob/master/src/python/coneprog.py
    if status == :Optimal
        @test pobj ≈ dobj atol=atol rtol=rtol
        @test A*x ≈ b atol=atol rtol=rtol
        @test G*x + s ≈ h atol=atol rtol=rtol
        @test G'*z + A'*y ≈ -c atol=atol rtol=rtol
        @test dot(s, z) ≈ 0.0 atol=atol rtol=rtol
        @test dot(c, x) ≈ pobj atol=1e-8 rtol=1e-8
        @test dot(b, y) + dot(h, z) ≈ -dobj atol=1e-8 rtol=1e-8
    elseif status == :PrimalInfeasible
        @test isnan(pobj)
        @test dobj ≈ 1.0 atol=1e-8 rtol=1e-8
        @test all(isnan, x)
        @test all(isnan, s)
        @test dot(b, y) + dot(h, z) ≈ -1.0 atol=1e-8 rtol=1e-8
        @test G'*z ≈ -A'*y atol=atol rtol=rtol
    elseif status == :DualInfeasible
        @test isnan(dobj)
        @test pobj ≈ -1.0 atol=1e-8 rtol=1e-8
        @test all(isnan, y)
        @test all(isnan, z)
        @test dot(c, x) ≈ -1.0 atol=1e-8 rtol=1e-8
        @test G*x ≈ -s atol=atol rtol=rtol
        @test A*x ≈ zeros(length(y)) atol=atol rtol=rtol
    elseif status == :IllPosed
        @test all(isnan, x)
        @test all(isnan, s)
        @test all(isnan, y)
        @test all(isnan, z)
    end

    stime = Hypatia.get_solvetime(mdl)
    niters = Hypatia.get_niters(mdl)

    return (x=x, y=y, s=s, z=z, pobj=pobj, dobj=dobj, status=status, stime=stime, niters=niters)
end


# native interface tests
include(joinpath(@__DIR__, "native.jl"))
@info("starting native interface tests")
verbose = false
lscachetypes = [
    Hypatia.QRSymmCache,
    Hypatia.NaiveCache,
    ]
testfuns = [
    _dimension1,
    _consistent1,
    _inconsistent1,
    _inconsistent2,
    _orthant1,
    _orthant2,
    _orthant3,
    _orthant4,
    _epinorminf1,
    _epinorminf2,
    _epinorminf3,
    _epinorminf4,
    _epinorminf5,
    _epinorminf6,
    _epinormeucl1,
    _epinormeucl2,
    _epipersquare1,
    _epipersquare2,
    _epipersquare3,
    _semidefinite1,
    _semidefinite2,
    _semidefinite3,
    _hypoperlog1,
    _hypoperlog2,
    _hypoperlog3,
    _hypoperlog4,
    _epiperpower1,
    _epiperpower2,
    _epiperpower3, # numerically unstable
    _hypogeomean1,
    _hypogeomean2,
    _hypogeomean3,
    _hypogeomean4,
    _epinormspectral1,
    _hypoperlogdet1,
    _hypoperlogdet2,
    _hypoperlogdet3,
    ]
@testset "native tests: $testfun, $lscachetype" for testfun in testfuns, lscachetype in lscachetypes
    testfun(verbose=verbose, lscachetype=lscachetype)
end


# examples in src/examples/ folder
egs_dir = joinpath(@__DIR__, "../examples")
include(joinpath(egs_dir, "envelope/envelope.jl"))
include(joinpath(egs_dir, "lp/lp.jl"))
include(joinpath(egs_dir, "namedpoly/namedpoly.jl"))

@info("starting default examples tests")
testfuns = [
    run_envelope,
    run_lp,
    run_namedpoly,
    ]
@testset "default examples: $testfun" for testfun in testfuns
    testfun()
end

include(joinpath(@__DIR__, "examples.jl"))
@info("starting varied examples tests")
verbose = false
lscachetypes = [
    Hypatia.QRSymmCache,
    # Hypatia.NaiveCache, # slow
    ]
testfuns = [
    _envelope1,
    _envelope2,
    _envelope3,
    _envelope4,
    _lp1,
    _lp2,
    _namedpoly1,
    _namedpoly2,
    _namedpoly3,
    # _namedpoly4, # interpolation memory usage excessive
    _namedpoly5,
    # _namedpoly6, # interpolation memory usage excessive
    _namedpoly7,
    _namedpoly8,
    _namedpoly9,
    # _namedpoly10, # numerically unstable
    _namedpoly11, # numerically unstable
    ]
@testset "varied examples: $testfun, $lscachetype" for testfun in testfuns, lscachetype in lscachetypes
    testfun(verbose=verbose, lscachetype=lscachetype)
end


# MathOptInterface tests
include(joinpath(@__DIR__, "moi.jl"))
@info("starting MathOptInterface tests")
verbose = false
lscachetypes = [
    Hypatia.QRSymmCache,
    Hypatia.NaiveCache,
    ]
@testset "MOI tests: $lscachetype, $(usedense ? "dense" : "sparse")" for lscachetype in lscachetypes, usedense in [false, true]
    testmoi(verbose=verbose, lscachetype=lscachetype, usedense=usedense)
end


return nothing
