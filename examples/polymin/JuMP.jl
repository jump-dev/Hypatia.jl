#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

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

function polyminJuMP(
    polyname::Symbol,
    halfdeg::Int;
    use_wsos::Bool = true,
    primal_wsos::Bool = false,
    sample::Bool = true,
    rseed::Int = 1,
    )
    (x, f, dom, true_obj) = getpolydata(polyname)

    if use_wsos
        Random.seed!(rseed)
        (U, pts, P0, PWts, _) = MU.interpolate(dom, halfdeg, sample = sample, sample_factor = 100)
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
        JuMP.@constraint(model, f >= a, domain = bss, maxdegree = 2 * halfdeg)
    end

    return (model = model, true_obj = true_obj)
end

polyminJuMP1() = polyminJuMP(:heart, 2)
polyminJuMP2() = polyminJuMP(:schwefel, 2)
polyminJuMP3() = polyminJuMP(:magnetism7_ball, 2)
polyminJuMP4() = polyminJuMP(:motzkin_ellipsoid, 4)
polyminJuMP5() = polyminJuMP(:caprasse, 4)
polyminJuMP6() = polyminJuMP(:goldsteinprice, 7)
polyminJuMP7() = polyminJuMP(:lotkavolterra, 3)
polyminJuMP8() = polyminJuMP(:robinson, 8)
polyminJuMP9() = polyminJuMP(:robinson_ball, 8)
polyminJuMP10() = polyminJuMP(:rosenbrock, 5)
polyminJuMP11() = polyminJuMP(:butcher, 2)
polyminJuMP12() = polyminJuMP(:goldsteinprice_ellipsoid, 7)
polyminJuMP13() = polyminJuMP(:goldsteinprice_ball, 7)
polyminJuMP14() = polyminJuMP(:motzkin, 3, primal_wsos = false)
polyminJuMP15() = polyminJuMP(:motzkin, 3)
polyminJuMP16() = polyminJuMP(:reactiondiffusion, 4, primal_wsos = false)
polyminJuMP17() = polyminJuMP(:lotkavolterra, 3, primal_wsos = false)
polyminJuMP18() = polyminJuMP(:heart, 2, use_wsos = false)
polyminJuMP19() = polyminJuMP(:schwefel, 2, use_wsos = false)
polyminJuMP20() = polyminJuMP(:magnetism7_ball, 2, use_wsos = false)
polyminJuMP21() = polyminJuMP(:motzkin_ellipsoid, 4, use_wsos = false)
polyminJuMP22() = polyminJuMP(:caprasse, 4, use_wsos = false)

function test_polyminJuMP(instance::Function; options)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.objective_value(d.model) ≈ d.true_obj atol = 1e-4 rtol = 1e-4
    return
end

test_polyminJuMP(; options...) = test_polyminJuMP.([
    polyminJuMP1,
    polyminJuMP2,
    polyminJuMP3,
    polyminJuMP4,
    polyminJuMP5,
    polyminJuMP6,
    polyminJuMP7,
    polyminJuMP8,
    polyminJuMP9,
    polyminJuMP10,
    polyminJuMP11,
    polyminJuMP12,
    polyminJuMP13,
    polyminJuMP14,
    polyminJuMP15,
    polyminJuMP16,
    polyminJuMP17,
    polyminJuMP18,
    polyminJuMP19,
    polyminJuMP20,
    polyminJuMP21,
    polyminJuMP22,
    ], options = options)

test_polyminJuMP_quick(; options...) = test_polyminJuMP.([
    polyminJuMP2,
    polyminJuMP3,
    polyminJuMP6,
    polyminJuMP14,
    polyminJuMP15,
    polyminJuMP20,
    ], options = options)
