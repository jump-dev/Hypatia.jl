#=
Copyright 2018, Chris Coey and contributors

see description in examples/polymin/native.jl
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
import SemialgebraicSets
import Random
using Test

include("polymindata.jl")

function build_JuMP_polymin_PSD(
    x,
    f,
    dom::MU.Domain;
    d::Int = div(max_degree(f) + 1, 2),
    )
    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-7, tol_rel_opt = 1e-7, tol_abs_opt = 1e-7))
    JuMP.@variable(model, a)
    JuMP.@objective(model, Max, a)
    bss = MU.get_domain_inequalities(dom, x)
    JuMP.@constraint(model, f >= a, domain = bss, maxdegree = 2d)

    return model
end

function build_JuMP_polymin_WSOS(
    x,
    f,
    dom::MU.Domain;
    d::Int = div(DynamicPolynomials.maxdegree(f) + 1, 2),
    sample::Bool = true,
    primal_wsos::Bool = false,
    rseed::Int = 1,
    )
    Random.seed!(rseed)
    n = DynamicPolynomials.nvariables(f)
    (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = sample, sample_factor = 100)
    cone = HYP.WSOSPolyInterpCone(U, [P0, PWts...], !primal_wsos)

    # build JuMP model
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-8, tol_rel_opt = 1e-7, tol_abs_opt = 1e-8))
    coeffs = [f(x => pts[j, :]) for j in 1:U]
    if primal_wsos
        JuMP.@variable(model, a)
        JuMP.@objective(model, Max, a)
        JuMP.@constraint(model, coeffs - a in cone)
    else
        JuMP.@variable(model, μ[1:U])
        JuMP.@objective(model, Min, sum(μ[j] * coeffs[j] for j in 1:U))
        JuMP.@constraint(model, sum(μ) == 1.0) # TODO can remove this constraint and a variable
        JuMP.@constraint(model, μ in cone)
    end

    return model
end

function polyminj(polyname::Symbol, d::Int; use_wsos::Bool = true, primal_wsos::Bool = false)
    (x, f, dom, true_obj) = getpolydata(polyname)
    if use_wsos
        model = build_JuMP_polymin_WSOS(x, f, dom, d = d, primal_wsos = primal_wsos)
    else
        model = build_JuMP_polymin_PSD(x, f, dom, d = d)
    end
    return (model = model, true_obj = true_obj)
end

polymin1j() = polyminj(:heart, 2)
polymin2j() = polyminj(:schwefel, 2)
polymin3j() = polyminj(:magnetism7_ball, 2)
polymin4j() = polyminj(:motzkin_ellipsoid, 4)
polymin5j() = polyminj(:caprasse, 4)
polymin6j() = polyminj(:goldsteinprice, 7)
polymin7j() = polyminj(:lotkavolterra, 3)
polymin8j() = polyminj(:robinson, 8)
polymin9j() = polyminj(:reactiondiffusion_ball, 3)
polymin10j() = polyminj(:rosenbrock, 5)
polymin11j() = polyminj(:butcher, 2)
polymin12j() = polyminj(:butcher_ball, 2)
polymin13j() = polyminj(:butcher_ellipsoid, 2)
polymin14j() = polyminj(:motzkin, 3, use_wsos = false)
polymin15j() = polyminj(:motzkin, 3, primal_wsos = true)
# TODO add more from dictionary

function test_polyminj(instance)
    (model, true_obj) = instance()

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
    @test primal_obj ≈ true_obj atol = 1e-4 rtol = 1e-4

    return
end

test_polyminj_many() = test_polyminj.([
    polymin1j,
    polymin2j,
    polymin3j,
    polymin4j,
    polymin5j,
    polymin6j,
    polymin7j,
    polymin8j,
    polymin9j,
    polymin10j,
    polymin11j,
    polymin12j,
    polymin13j,
    polymin14j,
    polymin15j,
])

test_polyminj_small() = test_polyminj.([
    polymin2j,
    polymin3j,
    polymin6j,
    polymin14j,
    polymin15j,
])
