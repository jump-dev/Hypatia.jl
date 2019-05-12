#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyOpt.m
formulates and solves the (dual of the) polynomial optimization problem for a given polynomial, described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MO = HYP.Models
const SO = HYP.Solvers
const MU = HYP.ModelUtilities

using LinearAlgebra
using Test

include("data.jl")

function build_polymin(
    n::Int,
    fn::Polynomial{true,Int64},
    dom::MU.Domain,
    deg::Int;
    primal_wsos::Bool = true,
    )
    # only works for boxes
    lbs = dom.l
    ubs = dom.u
    d = div(deg + 1, 2)

    # TODO choose which cone definition to use and cleanup below
    # generate interpolation
    # (U, pts, P0, _, _) = MU.wsos_box_params(n, d, false)
    dom = MU.Box(lbs, ubs)
    (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = (n >= 5))

    # TODO algorithm may perform better if function evaluations are rescaled to have more reasonable norm
    # set up problem data
    if primal_wsos
        c = [-1.0]
        A = zeros(0, 1)
        b = Float64[]
        G = ones(U, 1)
        h = [fn(pts[j, :]...) for j in 1:U]
    else
        c = [fn(pts[j, :]...) for j in 1:U] # evaluate polynomial at transformed points
        A = ones(1, U) # TODO eliminate constraint and first variable
        b = [1.0]
        G = Diagonal(-1.0I, U) # TODO uniformscaling?
        h = zeros(U)
    end
    cones = [CO.WSOSPolyInterp(U, [P0, PWts...], !primal_wsos)]
    # Ls = Int[size(P0, 2)]
    # @assert Ls[1] == binomial(n + d, n)
    # gs = Vector{Float64}[ones(U)]
    # for i in 1:n
    #     # Li = size(PWts[i], 2) # TODO may be wrong
    #     di = d - 1 # degree of gi is 2
    #     Li = binomial(n + di, n)
    #     gi = [(-pts[u, i] + ubs[i]) * (pts[u, i] - lbs[i]) for u in 1:U]
    #     push!(Ls, Li)
    #     push!(gs, gi)
    # end
    # cones = [CO.WSOSPolyInterp_2(U, P0, Ls, gs, !primal_wsos)]
    cone_idxs = [1:U]

    return (c, A, b, G, h, cones, cone_idxs)
end

function polymin(polyname::Symbol, deg::Int; primal_wsos::Bool = true)
    (x, fn, dom, true_obj) = getpolydata(polyname)
    model = build_polymin(length(x), fn, dom, deg, primal_wsos = primal_wsos)
    if primal_wsos
        true_obj *= -1
    end
    return (model = model, true_obj = true_obj)
end

polymin1() = polymin(:butcher, 2)
polymin2() = polymin(:caprasse, 4)
polymin3() = polymin(:goldsteinprice, 6)
polymin4() = polymin(:heart, 2)
polymin5() = polymin(:lotkavolterra, 3)
polymin6() = polymin(:magnetism7, 2)
polymin7() = polymin(:motzkin, 7)
polymin8() = polymin(:reactiondiffusion, 4)
polymin9() = polymin(:robinson, 8)
polymin10() = polymin(:rosenbrock, 5)
polymin11() = polymin(:schwefel, 4)
polymin12() = polymin(:reactiondiffusion, 4, primal_wsos = false)

function test_polymin(instance::Function)
    ((c, A, b, G, h, cones, cone_idxs), true_obj) = instance()
    model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = true)
    SO.solve(solver)
    r = SO.test_certificates(solver, model, atol = 1e-4, rtol = 1e-4)
    @test r.status == :Optimal
    @test r.primal_obj ≈ true_obj atol = 1e-4 rtol = 1e-4

    return
end

test_polymins() = test_polymin.([
    polymin1,
    polymin2,
    polymin3,
    polymin4,
    polymin5,
    polymin6,
    polymin7,
    polymin8,
    polymin9,
    polymin10,
    polymin11,
    polymin12,
    ])
