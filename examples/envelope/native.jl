#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the (dual of the) polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MU = HYP.ModelUtilities
const MO = HYP.Models

using LinearAlgebra
using SparseArrays
using DelimitedFiles
import Random
using Test

function build_envelope(
    npoly::Int,
    deg::Int,
    n::Int,
    d::Int;
    primal_wsos::Bool = true,
    use_data::Bool = false,
    dense::Bool = false,
    rseed::Int = 1,
    )
    Random.seed!(rseed)

    # generate interpolation
    @assert deg <= d
    domain = MU.Box(-ones(n), ones(n))
    (U, pts, P0, PWts, w) = MU.interpolate(domain, d, sample = false, calc_w = true)

    if use_data
        # use provided data in data folder
        c_or_h = vec(readdlm(joinpath(@__DIR__, "data/c$(npoly * U).txt"), ',', Float64))
    else
        # generate random data
        LDegs = binomial(n + deg, n)
        c_or_h = vec(P0[:, 1:LDegs] * rand(-9:9, LDegs, npoly))
    end

    subI = dense ? Array(1.0I, U, U) : sparse(1.0I, U, U)
    if primal_wsos
        # use formulation with WSOS cone in primal
        c = -w
        A = zeros(0, U)
        b = Float64[]
        G = repeat(subI, outer = (npoly, 1))
        h = c_or_h
    else
        # use formulation with WSOS cone in dual
        c = c_or_h
        A = repeat(subI, outer = (1, npoly))
        b = w
        G = Diagonal(-1.0I, npoly * U) # TODO uniformscaling
        h = zeros(npoly * U)
    end

    cones = [CO.WSOSPolyInterp(U, [P0, PWts...], !primal_wsos) for k in 1:npoly]
    cone_idxs = [(1 + (k - 1) * U):(k * U) for k in 1:npoly]

    return (c, A, b, G, h, cones, cone_idxs)
end

function run_envelope(primal_wsos::Bool, dense::Bool)
    # optionally use fixed data in folder
    # select number of polynomials and degrees for the envelope
    # select dimension and SOS degree (to be squared)
    (c, A, b, G, h, cones, cone_idxs) =
        # build_envelope(2, 5, 1, 5, use_data = true, primal_wsos = primal_wsos, dense = dense)
        # build_envelope(2, 5, 2, 6, primal_wsos = primal_wsos, dense = dense)
        build_envelope(3, 5, 3, 5, primal_wsos = primal_wsos, dense = dense)
        # build_envelope(2, 30, 1, 30, primal_wsos = primal_wsos, dense = dense)

    model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = true)
    SO.solve(solver)
    @test SO.get_status(solver) == :Optimal

    return
end

run_envelope_primal_dense() = run_envelope(true, true)
run_envelope_dual_dense() = run_envelope(false, true)
run_envelope_primal_sparse() = run_envelope(true, false)
run_envelope_dual_sparse() = run_envelope(false, false)
