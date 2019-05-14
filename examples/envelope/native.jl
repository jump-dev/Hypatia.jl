#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the (dual of the) polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using LinearAlgebra
using SparseArrays
import Random
using Test
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MU = HYP.ModelUtilities

function envelope(
    npoly::Int,
    deg::Int,
    n::Int,
    d::Int;
    primal_wsos::Bool = true,
    use_dense::Bool = false,
    )
    # generate interpolation
    @assert deg <= d
    domain = MU.Box(-ones(n), ones(n))
    (U, pts, P0, PWts, w) = MU.interpolate(domain, d, sample = false, calc_w = true)

    # generate random data
    LDegs = binomial(n + deg, n)
    c_or_h = vec(P0[:, 1:LDegs] * rand(-9:9, LDegs, npoly))

    subI = use_dense ? Array(1.0I, U, U) : sparse(1.0I, U, U)
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

    return (model = (c, A, b, G, h, cones, cone_idxs),)
end

envelope1(; primal_wsos::Bool = true, use_dense::Bool = true) = envelope(2, 5, 2, 6, primal_wsos = primal_wsos, use_dense = use_dense)
envelope2(; primal_wsos::Bool = true, use_dense::Bool = true) = envelope(3, 5, 3, 5, primal_wsos = primal_wsos, use_dense = use_dense)
envelope3(; primal_wsos::Bool = true, use_dense::Bool = true) = envelope(2, 30, 1, 30, primal_wsos = primal_wsos, use_dense = use_dense)

function test_envelope(instance::Function; rseed::Int = 1)
    Random.seed!(rseed)
    ((c, A, b, G, h, cones, cone_idxs),) = instance()
    model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = true)
    SO.solve(solver)
    r = SO.test_certificates(solver, model, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    return
end

test_envelope() = test_envelope.([
    envelope1,
    envelope2,
    envelope3,
    ])
