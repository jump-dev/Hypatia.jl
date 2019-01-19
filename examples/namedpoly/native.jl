#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyOpt.m
formulates and solves the (dual of the) polynomial optimization problem for a given polynomial, described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const LS = HYP.LinearSystems
const MU = HYP.ModelUtilities

using LinearAlgebra
using Test

include("polydata.jl") # contains predefined polynomials

function build_namedpoly(
    polyname::Symbol,
    d::Int;
    )
    # get data for named polynomial
    (n, lbs, ubs, deg, fn) = polys[polyname]
    @assert d >= div(deg + 1, 2)

    # generate interpolation; use random sampling if n is large
    dom = MU.Box(lbs, ubs)
    (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample=(n >= 5))

    # set up problem data
    A = ones(1, U)
    b = [1.0,]
    c = [fn(pts[j,:]...) for j in 1:U] # evaluate polynomial at transformed points
    G = Diagonal(-1.0I, U) # TODO uniformscaling?
    h = zeros(U)

    cone = CO.Cone([CO.WSOSPolyInterp(U, [P0, PWts...], true)], [1:U])

    return (c, A, b, G, h, cone)
end

function run_namedpoly()
    # select the named polynomial to minimize and the SOS degree (to be squared)
    (c, A, b, G, h, cone) =
        # build_namedpoly(:butcher, 2)
        # build_namedpoly(:caprasse, 4)
        # build_namedpoly(:goldsteinprice, 7)
        # build_namedpoly(:heart, 2)
        # build_namedpoly(:lotkavolterra, 3)
        # build_namedpoly(:magnetism7, 2)
        # build_namedpoly(:motzkin, 7)
        build_namedpoly(:reactiondiffusion, 4)
        # build_namedpoly(:robinson, 8)
        # build_namedpoly(:rosenbrock, 4)
        # build_namedpoly(:schwefel, 3)

    HYP.check_data(c, A, b, G, h, cone)
    (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = HYP.preprocess_data(c, A, b, G, useQR=true)
    L = LS.QRSymm(c1, A1, b1, G1, h, cone, Q2, RiQ1)

    mdl = HYP.Model(maxiter=200, verbose=false)
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
