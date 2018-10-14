#=
Copyright 2018, Chris Coey and contributors
=#

using Hypatia
using Test
using Random
using LinearAlgebra
using SparseArrays


# TODO make first part a native interface function eventually
# TODO maybe build a new high-level optimizer struct. the current optimizer struct is low-level
function fullsolve(opt::Hypatia.Optimizer, c, A, b, G, h, cone, lscachetype)
    Hypatia.check_data(c, A, b, G, h, cone)
    (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)

    # TODO handle lscachetype
    if lscachetype != Hypatia.QRSymmCache
        error("tests only support QRSymmCache now")
    end
    L = Hypatia.QRSymmCache(c1, A1, b1, G1, h, cone, Q2, RiQ1)
    # L = Hypatia.NaiveCache(c1, A1, b1, G1, h, cone)

    Hypatia.load_data!(opt, c1, A1, b1, G1, h, cone, L)

    Hypatia.solve!(opt)

    x = zeros(length(c))
    x[dukeep] = Hypatia.get_x(opt)
    y = zeros(length(b))
    y[prkeep] = Hypatia.get_y(opt)
    s = Hypatia.get_s(opt)
    z = Hypatia.get_z(opt)

    pobj = Hypatia.get_pobj(opt)
    dobj = Hypatia.get_dobj(opt)

    status = Hypatia.get_status(opt)
    stime = Hypatia.get_solvetime(opt)
    niters = Hypatia.get_niters(opt)

    return (x=x, y=y, s=s, z=z, pobj=pobj, dobj=dobj, status=status, stime=stime, niters=niters)
end


# native interface tests
include(joinpath(@__DIR__, "native.jl"))
verbose = false # test verbosity
lscachetype = Hypatia.QRSymmCache # linear system cache type

@testset "native interface tests" begin
    for testfun in (
        _consistent1,
        _inconsistent1,
        _inconsistent2,
        _orthant1,
        _orthant2,
        _orthant3,
        _orthant4,
        _ellinf1,
        _ellinf2,
        _ellinfdual1,
        _ellinfdual2,
        _soc1,
        _rsoc1,
        _rsoc2,
        _psd1,
        _psd2,
        _exp1,
        _power1,
        )
        testfun(verbose, lscachetype)
    end
end

# examples in src/examples/ folder
egs_dir = joinpath(@__DIR__, "../examples")
include(joinpath(egs_dir, "envelope/envelope.jl"))
include(joinpath(egs_dir, "lp/lp.jl"))
include(joinpath(egs_dir, "namedpoly/namedpoly.jl"))

@testset "default examples" begin
    run_envelope()
    run_lp()
    run_namedpoly()
end

include(joinpath(@__DIR__, "examples.jl"))
verbose = false # test verbosity
lscachetype = Hypatia.QRSymmCache # linear system cache type

@testset "varied examples" begin
    for testfun in (
        _envelope1,
        _envelope2,
        _envelope3,
        # _envelope4,
        _lp1,
        _lp2,
        _lp3,
        _namedpoly1,
        _namedpoly2,
        _namedpoly3,
        # _namedpoly4,
        _namedpoly5,
        # _namedpoly6,
        _namedpoly7,
        _namedpoly8,
        _namedpoly9,
        _namedpoly10,
        _namedpoly11,
        )
        testfun(verbose, lscachetype)
    end
end


# MathOptInterface tests
verbose = false # test verbosity
include(joinpath(@__DIR__, "moi.jl"))
@testset "MathOptInterface tests" begin
    testmoi(verbose, false)
    testmoi(verbose, true)
end


return nothing
