#=
Copyright 2018, Chris Coey and contributors

see description in examples/namedpoly/native.jl
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MU = HYP.ModelUtilities

import MathOptInterface
const MOI = MathOptInterface
import JuMP
import MultivariatePolynomials
import DynamicPolynomials
import SumOfSquares
import PolyJuMP
import Random
using Test

# list of predefined polynomials from various applications
# see https://people.sc.fsu.edu/~jburkardt/py_src/polynomials/polynomials.html
function getpolydata(polyname::Symbol)
    if polyname == :butcher
        DynamicPolynomials.@polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        dom = MU.Box([-1,-0.1,-0.1,-1,-0.1,-0.1], [0,0.9,0.5,-0.1,-0.05,-0.03])
        truemin = -1.4393333333
    elseif polyname == :butcher_ball
        DynamicPolynomials.@polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        axes = 0.5 * ([0,0.9,0.5,-0.1,-0.05,-0.03] - [-1,-0.1,-0.1,-1,-0.1,-0.1])
        centers = 0.5 * ([-1,-0.1,-0.1,-1,-0.1,-0.1] + [0,0.9,0.5,-0.1,-0.05,-0.03])
        dom = MU.Ball(centers, sqrt(6) * maximum(axes))
        truemin = -4.10380
    elseif polyname == :butcher_ellipsoid
        DynamicPolynomials.@polyvar x[1:6]
        f = x[6]*x[2]^2+x[5]*x[3]^2-x[1]*x[4]^2+x[4]^3+x[4]^2-1/3*x[1]+4/3*x[4]
        # heuristically-obtained enclosing ellipsoid
        centers = 0.5 * ([-1,-0.1,-0.1,-1,-0.1,-0.1] + [0,0.9,0.5,-0.1,-0.05,-0.03])
        Q = Diagonal(6 * abs2.(centers))
        dom = MU.Ellipsoid(centers, Q)
        truemin = -16.7378208
    elseif polyname == :caprasse
        DynamicPolynomials.@polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        dom = MU.Box(-0.5 * ones(4), 0.5 * ones(4))
        truemin = -3.1800966258
    elseif polyname == :caprasse_ball
        DynamicPolynomials.@polyvar x[1:4]
        f = -x[1]*x[3]^3+4x[2]*x[3]^2*x[4]+4x[1]*x[3]*x[4]^2+2x[2]*x[4]^3+4x[1]*x[3]+4x[3]^2-10x[2]*x[4]-10x[4]^2+2
        dom = MU.Ball(zeros(4), 1.0)
        truemin = -9.47843346
    elseif polyname == :goldsteinprice
        DynamicPolynomials.@polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = MU.Box(-2 * ones(2), 2 * ones(2))
        truemin = 3
    elseif polyname == :goldsteinprice_ball
        DynamicPolynomials.@polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        dom = MU.Ball(zeros(2), 2*sqrt(2))
        truemin = 3
    elseif polyname == :goldsteinprice_ellipsoid
        DynamicPolynomials.@polyvar x[1:2]
        f = (1+(x[1]+x[2]+1)^2*(19-14x[1]+3x[1]^2-14x[2]+6x[1]*x[2]+3x[2]^2))*(30+(2x[1]-3x[2])^2*(18-32x[1]+12x[1]^2+48x[2]-36x[1]*x[2]+27x[2]^2))
        centers = zeros(2)
        Q = Diagonal(0.25 * ones(2))
        dom = MU.Ellipsoid(centers, Q)
        truemin = 3
    elseif polyname == :heart
        DynamicPolynomials.@polyvar x[1:8]
        f = x[1]*x[6]^3-3x[1]*x[6]*x[7]^2+x[3]*x[7]^3-3x[3]*x[7]*x[6]^2+x[2]*x[5]^3-3*x[2]*x[5]*x[8]^2+x[4]*x[8]^3-3x[4]*x[8]*x[5]^2+0.9563453
        dom = MU.Box([-0.1,0.4,-0.7,-0.7,0.1,-0.1,-0.3,-1.1], [0.4,1,-0.4,0.4,0.2,0.2,1.1,-0.3])
        truemin = -1.36775
    elseif polyname == :lotkavolterra
        DynamicPolynomials.@polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = MU.Box(-2 * ones(4), 2 * ones(4))
        truemin = -20.8
    elseif polyname == :lotkavolterra_ball
        DynamicPolynomials.@polyvar x[1:4]
        f = x[1]*(x[2]^2+x[3]^2+x[4]^2-1.1)+1
        dom = MU.Ball(zeros(4), 4.0)
        truemin = -21.13744
    elseif polyname == :magnetism7
        DynamicPolynomials.@polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = MU.Box(-ones(7), ones(7))
        truemin = -0.25
    elseif polyname == :magnetism7_ball
        DynamicPolynomials.@polyvar x[1:7]
        f = x[1]^2+2x[2]^2+2x[3]^2+2x[4]^2+2x[5]^2+2x[6]^2+2x[7]^2-x[1]
        dom = MU.Ball(zeros(7), sqrt(7))
        truemin = -0.25
    elseif polyname == :motzkin
        DynamicPolynomials.@polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = MU.Box(-ones(2), ones(2))
        truemin = 0
    elseif polyname == :motzkin_ball
        DynamicPolynomials.@polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        dom = MU.Ball(zeros(2), sqrt(2))
        truemin = 0
    elseif polyname == :motzkin_ellipsoid
        # ellipsoid contains two local minima in opposite orthants
        DynamicPolynomials.@polyvar x[1:2]
        f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
        Q = [1 1; 1 -1]
        D = [1 0; 0 0.1]
        S = Q * D * Q
        dom = MU.Ellipsoid(zeros(2), S)
        truemin = 0
    elseif polyname == :reactiondiffusion
        DynamicPolynomials.@polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = MU.Box(-5 * ones(3), 5 * ones(3))
        truemin = -36.71269068
    elseif polyname == :reactiondiffusion_ball
        DynamicPolynomials.@polyvar x[1:3]
        f = -x[1]+2x[2]-x[3]-0.835634534x[2]*(1+x[2])
        dom = MU.Ball(zeros(3), 5*sqrt(3))
        truemin = -73.31
    elseif polyname == :robinson
        DynamicPolynomials.@polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = MU.Box(-ones(2), ones(2))
        truemin = 0.814814
    elseif polyname == :robinson_ball
        DynamicPolynomials.@polyvar x[1:2]
        f = 1+x[1]^6+x[2]^6-x[1]^4*x[2]^2+x[1]^4-x[1]^2*x[2]^4+x[2]^4-x[1]^2+x[2]^2+3x[1]^2*x[2]^2
        dom = MU.Ball(zeros(2), sqrt(2))
        truemin = 0.814814
    elseif polyname == :rosenbrock
        DynamicPolynomials.@polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = MU.Box(-5 * ones(2), 10 * ones(2))
        truemin = 0
    elseif polyname == :rosenbrock_ball
        DynamicPolynomials.@polyvar x[1:2]
        f = (1-x[1])^2+100*(x[1]^2-x[2])^2
        dom = MU.Ball(2.5 * ones(2), 7.5*sqrt(2))
        truemin = 0
    elseif polyname == :schwefel
        DynamicPolynomials.@polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = MU.Box(-10 * ones(3), 10 * ones(3))
        truemin = 0
    elseif polyname == :schwefel_ball
        DynamicPolynomials.@polyvar x[1:3]
        f = (x[1]-x[2]^2)^2+(x[2]-1)^2+(x[1]-x[3]^2)^2+(x[3]-1)^2
        dom = MU.Ball(zeros(3), 10*sqrt(3))
        truemin = 0
    else
        error("poly $polyname not recognized")
    end

    return (x, f, dom, truemin)
end

function build_JuMP_namedpoly_PSD(
    x,
    f,
    dom::MU.Domain;
    d::Int = div(max_degree(f) + 1, 2),
    )
    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-9, tol_rel_opt = 1e-8, tol_abs_opt = 1e-8))
    JuMP.@variable(model, a)
    JuMP.@objective(model, Max, a)
    JuMP.@constraint(model, fnn, f >= a, domain = MU.get_domain_inequalities(dom, x), max_degree = 2d)

    return model
end

function build_JuMP_namedpoly_WSOS(
    x,
    f,
    dom::MU.Domain;
    d::Int = div(DynamicPolynomials.maxdegree(f) + 1, 2),
    sample::Bool = true,
    primal_wsos::Bool = true,
    rseed::Int = 1,
    )
    Random.seed!(rseed)

    (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = sample, sample_factor = 100)

    # build JuMP model
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-9, tol_rel_opt = 1e-8, tol_abs_opt = 1e-8))
    if primal_wsos
        JuMP.@variable(model, a)
        JuMP.@objective(model, Max, a)
        JuMP.@constraint(model, [f(pts[j, :]) - a for j in 1:U] in HYP.WSOSPolyInterpCone(U, [P0, PWts...], false))
    else
        JuMP.@variable(model, x[1:U])
        JuMP.@objective(model, Min, sum(x[j] * f(pts[j, :]...) for j in 1:U))
        JuMP.@constraint(model, sum(x) == 1.0)
        JuMP.@constraint(model, x in HYP.WSOSPolyInterpCone(U, [P0, PWts...], true))
    end

    return model
end

function run_JuMP_namedpoly(use_wsos::Bool; primal_wsos::Bool = false)
    # select the named polynomial to minimize and degree of SOS interpolation
    (polyname, deg) =
        # :butcher, 2
        # :butcher_ball, 2
        # :butcher_ellipsoid, 2
        # :caprasse, 4
        # :caprasse_ball, 4
        # :goldsteinprice, 7
        # :goldsteinprice_ball, 7
        # :goldsteinprice_ellipsoid, 7
        # :heart, 2
        # :lotkavolterra, 3
        # :lotkavolterra_ball, 3
        # :magnetism7, 2
        # :magnetism7_ball, 2
        # :motzkin, 7
        # :motzkin_ball, 7
        # :motzkin_ellipsoid, 7
        # :reactiondiffusion, 3
        :reactiondiffusion_ball, 3
        # :robinson, 8
        # :robinson_ball, 8
        # :rosenbrock, 4
        # :rosenbrock_ball, 4
        # :schwefel, 3
        # :schwefel_ball, 3

    (x, f, dom, truemin) = getpolydata(polyname)

    if use_wsos
        model = build_JuMP_namedpoly_WSOS(x, f, dom, d = deg, primal_wsos = primal_wsos)
    else
        model = build_JuMP_namedpoly_PSD(x, f, dom, d = deg)
    end
    JuMP.optimize!(model)

    term_status = JuMP.termination_status(model)
    primal_obj = JuMP.objective_value(model)
    dual_obj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.OPTIMAL
    @test pr_status == MOI.FEASIBLE_POINT
    @test du_status == MOI.FEASIBLE_POINT
    @test primal_obj ≈ dual_obj atol = 1e-4 rtol = 1e-4
    @test primal_obj ≈ truemin atol = 1e-4 rtol = 1e-4

    return
end

run_JuMP_namedpoly_PSD() = run_JuMP_namedpoly(false)
run_JuMP_namedpoly_WSOS_primal() = run_JuMP_namedpoly(true, primal_wsos = true)
run_JuMP_namedpoly_WSOS_dual() = run_JuMP_namedpoly(true, primal_wsos = false)
