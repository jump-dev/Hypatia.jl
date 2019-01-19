#=
Copyright 2018, Chris Coey and contributors

see description in examples/namedpoly/native.jl
=#

using Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const LS = HYP.LinearSystems
const MU = HYP.ModelUtilities

import MathOptInterface
MOI = MathOptInterface
import JuMP
import MultivariatePolynomials
import DynamicPolynomials
import SumOfSquares
import PolyJuMP
using Random
using Test

include("polydata.jl") # contains predefined polynomials

function build_JuMP_namedpoly_PSD(
    x,
    f,
    dom::MU.Domain;
    d::Int = div(maxdegree(f) + 1, 2),
    )
    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose=true))
    JuMP.@variable(model, a)
    JuMP.@objective(model, Max, a)
    JuMP.@constraint(model, fnn, f >= a, domain=MU.get_domain_inequalities(dom, x), maxdegree=2d)

    return model
end

function build_JuMP_namedpoly_WSOS(
    x,
    f,
    dom::MU.Domain;
    d::Int = div(DynamicPolynomials.maxdegree(f) + 1, 2),
    sample_factor = 25,
    primal_wsos::Bool = false,
    rseed::Int = 1,
    )
    (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample=true)

    # build JuMP model
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose=true, tolrelopt=1e-7, tolfeas=1e-8))
    if primal_wsos
        JuMP.@variable(model, a)
        JuMP.@objective(model, Max, a)
        JuMP.@constraint(model, [f(pts[j,:]) - a for j in 1:U] in HYP.WSOSPolyInterpCone(U, [P0, PWts...], false))
    else
        JuMP.@variable(model, x[1:U])
        JuMP.@objective(model, Min, sum(x[j] * f(pts[j,:]...) for j in 1:U))
        JuMP.@constraint(model, sum(x) == 1.0)
        JuMP.@constraint(model, x in HYP.WSOSPolyInterpCone(U, [P0, PWts...], true))
    end

    return model
end

function run_JuMP_namedpoly(use_wsos::Bool; primal_wsos::Bool=false)
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
        model = build_JuMP_namedpoly_WSOS(x, f, dom, d=deg, primal_wsos=primal_wsos)
    else
        model = build_JuMP_namedpoly_PSD(x, f, dom, d=deg)
    end
    JuMP.optimize!(model)

    term_status = JuMP.termination_status(model)
    pobj = JuMP.objective_value(model)
    dobj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.OPTIMAL
    @test pr_status == MOI.FEASIBLE_POINT
    @test du_status == MOI.FEASIBLE_POINT
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4

    return
end

run_JuMP_namedpoly_PSD() = run_JuMP_namedpoly(false)
run_JuMP_namedpoly_WSOS_primal() = run_JuMP_namedpoly(true, primal_wsos=true)
run_JuMP_namedpoly_WSOS_dual() = run_JuMP_namedpoly(true, primal_wsos=false)
