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

# list of predefined polynomials and domains from various applications
# see https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html
polys = Dict{Symbol, NamedTuple}(
    :butcher => (n=6, lbs=[-1.0,-0.1,-0.1,-1.0,-0.1,-0.1], ubs=[0.0,0.9,0.5,-0.1,-0.05,-0.03], deg=3,
        fn=((u,v,w,x,y,z) -> z*v^2+y*w^2-u*x^2+x^3+x^2-(1/3)*u+(4/3)*x)
        ),
    :caprasse => (n=4, lbs=-0.5 * ones(4), ubs=0.5 * ones(4), deg=8,
        fn=((w,x,y,z) -> -w*y^3+4x*y^2*z+4w*y*z^2+2x*z^3+4w*y+4y^2-10x*z-10z^2+2)
        ),
    :goldsteinprice => (n=2, lbs=-2.0 * ones(2), ubs=2.0 * ones(2), deg=8,
        fn=((x,y) -> (1+(x+y+1)^2*(19-14x+3x^2-14y+6x*y+3y^2))*(30+(2x-3y)^2*(18-32x+12x^2+48y-36x*y+27y^2)))
        ),
    :heart => (n=8, lbs=[-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], ubs=[0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3], deg=4,
        fn=((s,t,u,v,w,x,y,z) -> s*x^3-3s*x*y^2+u*y^3-3u*y*x^2+t*w^3-3*t*w*z^2+v*z^3-3v*z*w^2+0.9563453)
        ),
    :lotkavolterra => (n=4, lbs=-2.0 * ones(4), ubs=2.0 * ones(4), deg=3,
        fn=((w,x,y,z) -> w*(x^2+y^2+z^2-1.1)+1)
        ),
    :magnetism7 => (n=7, lbs=-ones(7), ubs=ones(7), deg=2,
        fn=((t,u,v,w,x,y,z) -> t^2+2u^2+2v^2+2w^2+2x^2+2y^2+2z^2-t)
        ),
    :motzkin => (n=2, lbs=-ones(2), ubs=ones(2), deg=6,
        fn=((x,y) -> 1-48x^2*y^2+64x^2*y^4+64x^4*y^2)
        ),
    :reactiondiffusion => (n=3, lbs=-5.0 * ones(3), ubs=5.0 * ones(3), deg=2,
        fn=((x,y,z) -> -x+2y-z-0.835634534y*(1+y))
        ),
    :robinson => (n=2, lbs=-ones(2), ubs=ones(2), deg=6,
        fn=((x,y) -> 1+x^6+y^6-x^4*y^2+x^4-x^2*y^4+y^4-x^2+y^2+3x^2*y^2)
        ),
    :rosenbrock => (n=2, lbs=-5.0 * ones(2), ubs=10.0 * ones(2), deg=4,
        fn=((x,y) -> (1-x)^2+100*(x^2-y)^2)
        ),
    :schwefel => (n=3, lbs=-10.0 * ones(3), ubs=10.0 * ones(3), deg=4,
        fn=((x,y,z) -> (x-y^2)^2+(y-1)^2+(x-z^2)^2+(z-1)^2)
        ),
)

function build_polymin(
    polyname::Symbol,
    d::Int;
    primal_wsos::Bool = true,
    )
    # get data for polynomial and domain
    (n, lbs, ubs, deg, fn) = polys[polyname]
    @assert d >= div(deg + 1, 2)

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

polymin1() = build_polymin(:butcher, 2)
polymin2() = build_polymin(:caprasse, 4)
polymin3() = build_polymin(:goldsteinprice, 6)
polymin4() = build_polymin(:heart, 2)
polymin5() = build_polymin(:lotkavolterra, 3)
polymin6() = build_polymin(:magnetism7, 2)
polymin7() = build_polymin(:motzkin, 7)
polymin8() = build_polymin(:reactiondiffusion, 4)
polymin9() = build_polymin(:robinson, 8)
polymin10() = build_polymin(:rosenbrock, 5)
polymin11() = build_polymin(:schwefel, 4)

function run_polymin()
    # select the polynomial/domain to minimize over and the SOS degree (to be squared)
    (c, A, b, G, h, cones, cone_idxs) = polymin8()
    model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = true)
    SO.solve(solver)
    @test SO.get_status(solver) == :Optimal

    return
end
