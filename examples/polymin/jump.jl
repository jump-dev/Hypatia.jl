#=
Copyright 2018, Chris Coey and contributors

see description in examples/polymin/native.jl
=#

import Random
using LinearAlgebra
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import SumOfSquares
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

include(joinpath(@__DIR__, "data.jl"))

function polymin_JuMP(
    polyname::Symbol,
    d::Int;
    use_wsos::Bool = true,
    primal_wsos::Bool = false,
    sample::Bool = true,
    rseed::Int = 1,
    )
    (x, f, dom, true_obj) = getpolydata(polyname)

    if use_wsos
        Random.seed!(rseed)
        (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = sample, sample_factor = 100)
        cone = HYP.WSOSPolyInterpCone(U, [P0, PWts...], !primal_wsos)
        interp_vals = [f(x => pts[j, :]) for j in 1:U]

        model = JuMP.Model()
        if primal_wsos
            JuMP.@variable(model, a)
            JuMP.@objective(model, Max, a)
            JuMP.@constraint(model, interp_vals .- a in cone)
        else
            JuMP.@variable(model, μ[1:U])
            JuMP.@objective(model, Min, dot(μ, interp_vals))
            JuMP.@constraint(model, sum(μ) == 1.0) # TODO can remove this constraint and a variable
            JuMP.@constraint(model, μ in cone)
        end
    else
        model = SumOfSquares.SOSModel()
        JuMP.@variable(model, a)
        JuMP.@objective(model, Max, a)
        bss = MU.get_domain_inequalities(dom, x)
        JuMP.@constraint(model, f >= a, domain = bss, maxdegree = 2d)
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

function test_polymin_JuMP(instance::Function; options)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.objective_value(d.model) ≈ d.true_obj atol = 1e-4 rtol = 1e-4
    return
end

test_polymin_JuMP(; options...) = test_polymin_JuMP.([
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
    ], options = options)

test_polymin_JuMP_quick(; options...) = test_polymin_JuMP.([
    polymin2_JuMP,
    polymin3_JuMP,
    polymin6_JuMP,
    polymin14_JuMP,
    polymin15_JuMP,
    ], options = options)
