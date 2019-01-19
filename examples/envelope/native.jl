#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the (dual of the) polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const LS = HYP.LinearSystems
const MU = HYP.ModelUtilities

using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Random
using Test

function build_envelope(
    npoly::Int,
    deg::Int,
    n::Int,
    d::Int;
    primal_wsos::Bool = true,
    use_data::Bool = false,
    usedense::Bool = false,
    rseed::Int = 1,
    )
    # generate interpolation
    @assert deg <= d
    domain = MU.Box(-ones(n), ones(n))
    (U, pts, P0, PWts, w) = MU.interpolate(domain, d, sample=false, calc_w=true)

    if use_data
        # use provided data in data folder
        c_or_h = vec(readdlm(joinpath(@__DIR__, "data/c$(npoly*U).txt"), ',', Float64))
    else
        # generate random data
        Random.seed!(rseed)
        LDegs = binomial(n+deg, n)
        c_or_h = vec(P0[:, 1:LDegs] * rand(-9:9, LDegs, npoly))
    end

    subI = usedense ? Array(1.0I, U, U) : sparse(1.0I, U, U)
    if primal_wsos
        # use formulation with WSOS cone in primal
        c = -w
        A = zeros(0, U)
        b = Float64[]
        G = repeat(subI, outer=(npoly, 1))
        h = c_or_h
    else
        # use formulation with WSOS cone in dual
        c = c_or_h
        A = repeat(subI, outer=(1, npoly))
        b = w
        G = Diagonal(-1.0I, npoly * U) # TODO uniformscaling
        h = zeros(npoly*U)
    end

    cone = CO.Cone(
        [CO.WSOSPolyInterp(U, [P0, PWts...], !primal_wsos) for k in 1:npoly],
        [(1 + (k - 1) * U):(k * U) for k in 1:npoly]
        )

    return (c, A, b, G, h, cone)
end

function run_envelope(primal_wsos::Bool, usedense::Bool)
    # optionally use fixed data in folder
    # select number of polynomials and degrees for the envelope
    # select dimension and SOS degree (to be squared)
    (c, A, b, G, h, cone) =
        # build_envelope(2, 5, 1, 5, use_data=true, primal_wsos=primal_wsos, usedense=usedense)
        build_envelope(2, 5, 2, 6, primal_wsos=primal_wsos, usedense=usedense)
        # build_envelope(3, 5, 3, 5, primal_wsos=primal_wsos, usedense=usedense)
        # build_envelope(2, 30, 1, 30, primal_wsos=primal_wsos, usedense=usedense)

    HYP.check_data(c, A, b, G, h, cone)
    (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = HYP.preprocess_data(c, A, b, G, useQR=true)
    L = LS.QRSymm(c1, A1, b1, G1, h, cone, Q2, RiQ1)

    mdl = HYP.Model(maxiter=200, verbose=true)
    HYP.load_data!(mdl, c1, A1, b1, G1, h, cone, L)
    HYP.solve!(mdl)

    x = zeros(length(c))
    x[dukeep] = HYP.get_x(mdl)
    y = zeros(length(b))
    y[prkeep] = HYP.get_y(mdl)
    s = HYP.get_s(mdl)
    z = HYP.get_z(mdl)

    status = HYP.get_status(mdl)
    solvetime = HYP.get_solvetime(mdl)
    pobj = HYP.get_pobj(mdl)
    dobj = HYP.get_dobj(mdl)

    @test status == :Optimal

    return
end

run_envelope_primal_dense() = run_envelope(true, true)
run_envelope_dual_dense() = run_envelope(false, true)
run_envelope_primal_sparse() = run_envelope(true, false)
run_envelope_dual_sparse() = run_envelope(false, false)
