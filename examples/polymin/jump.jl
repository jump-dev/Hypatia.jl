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

include("data.jl")

function build_polymin_JuMP_PSD(
    model::JuMP.Model,
    x,
    f,
    dom::MU.Domain;
    d::Int = div(max_degree(f) + 1, 2),
    )
    PolyJuMP.setpolymodule!(model, SumOfSquares)
    JuMP.@variable(model, a)
    JuMP.@objective(model, Max, a)
    bss = MU.get_domain_inequalities(dom, x)
    JuMP.@constraint(model, f >= a, domain = bss, maxdegree = 2d)

    return model
end

function build_polymin_JuMP_WSOS(
    model::JuMP.Model,
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

    coefs = [f(x => pts[j, :]) for j in 1:U]
    if primal_wsos
        JuMP.@variable(model, a)
        JuMP.@objective(model, Max, a)
        JuMP.@constraint(model, coefs .- a in cone)
    else
        JuMP.@variable(model, μ[1:U])
        JuMP.@objective(model, Min, sum(μ[j] * coefs[j] for j in 1:U))
        JuMP.@constraint(model, sum(μ) == 1.0) # TODO can remove this constraint and a variable
        JuMP.@constraint(model, μ in cone)
    end

    return model
end

function polymin_JuMP(polyname::Symbol, d::Int; use_wsos::Bool = true, primal_wsos::Bool = false)
    (x, f, dom, true_obj) = getpolydata(polyname)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-8, tol_rel_opt = 1e-7, tol_abs_opt = 1e-8))
    if use_wsos
        model = build_polymin_JuMP_WSOS(model, x, f, dom, d = d, primal_wsos = primal_wsos)
    else
        model = build_polymin_JuMP_PSD(model, x, f, dom, d = d)
    end
    return (model = model, true_obj = true_obj)
end

polymin1_JuMP() = polymin_JuMP(:heart, 2)
polymin2_JuMP() = polymin_JuMP(:schwefel, 2)
polymin3_JuMP() = polymin_JuMP(:magnetism7_ball, 2)
polymin4_JuMP() = polymin_JuMP(:motzkin_ellipsoid, 4)
polymin5_JuMP() = polymin_JuMP(:caprasse, 4)
polymin6_JuMP() = polymin_JuMP(:goldsteinprice, 7)
polymin7_JuMP() = polymin_JuMP(:lotkavolterra, 3)
polymin8_JuMP() = polymin_JuMP(:robinson, 8)
polymin9_JuMP() = polymin_JuMP(:reactiondiffusion_ball, 3)
polymin10_JuMP() = polymin_JuMP(:rosenbrock, 5)
polymin11_JuMP() = polymin_JuMP(:butcher, 2)
polymin12_JuMP() = polymin_JuMP(:butcher_ball, 2)
polymin13_JuMP() = polymin_JuMP(:butcher_ellipsoid, 2)
polymin14_JuMP() = polymin_JuMP(:motzkin, 3, use_wsos = false)
polymin15_JuMP() = polymin_JuMP(:motzkin, 3, primal_wsos = true)
# TODO add more from dictionary and more with different options combinations

function test_polymin_JuMP(instance::Function)
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

test_polymin_JuMP_many() = test_polymin_JuMP.([
    polymin1_JuMP,
    polymin2_JuMP,
    polymin3_JuMP,
    polymin4_JuMP,
    polymin5_JuMP,
    polymin6_JuMP,
    polymin7_JuMP,
    polymin8_JuMP,
    polymin9_JuMP,
    polymin10_JuMP,
    polymin11_JuMP,
    polymin12_JuMP,
    polymin13_JuMP,
    polymin14_JuMP,
    polymin15_JuMP,
    ])

test_polymin_JuMP_small() = test_polymin_JuMP.([
    polymin2_JuMP,
    polymin3_JuMP,
    polymin6_JuMP,
    polymin14_JuMP,
    polymin15_JuMP,
    ])
