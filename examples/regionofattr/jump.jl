#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

univariate cubic dynamical system
example taken from "Convex computation of the region of attraction of polynomial control systems" by D. Henrion and M. Korda
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const SO = HYP.Solvers
const MO = HYP.Models
const MU = HYP.ModelUtilities

import JuMP
import SumOfSquares
import SemialgebraicSets
const SAS = SemialgebraicSets
import MathOptInterface
const MOI = MathOptInterface
import PolyJuMP
import DynamicPolynomials
const DP = DynamicPolynomials
using LinearAlgebra
using Test

function regionofattr_JuMP(deg::Int; use_WSOS::Bool = true)
    T = 100.0
    DP.@polyvar x
    DP.@polyvar t
    f = x * (x - 0.5) * (x + 0.5) * T
    model = JuMP.Model()
    JuMP.@variables(model, begin
        v, PolyJuMP.Poly(DP.monomials([x; t], 0:deg))
        w, PolyJuMP.Poly(DP.monomials(x, 0:deg))
    end)
    dvdt = DP.differentiate(v, t) + DP.differentiate(v, x) * f
    diffwv = w - DP.subs(v, t => 0.0) - 1.0
    vT = DP.subs(v, t => 1.0)

    if use_WSOS
        dom1 = MU.Box([-1.0], [1.0]) # just state
        dom2 = MU.Box([-1.0, 0.0], [1.0, 1.0]) # state and time
        dom3 = MU.Box([-0.01], [0.01]) # state at the end
        (U1, pts1, P01, PWts1, quad_weights) = MU.interpolate(dom1, div(deg, 2) + 1, sample = false, calc_w = true)
        (U2, pts2, P02, PWts2, _) = MU.interpolate(dom2, div(deg, 2) + 1, sample = false)
        (U3, pts3, P03, PWts3, _) = MU.interpolate(dom3, div(deg, 2), sample = false)
        wsos_cone1 = HYP.WSOSPolyInterpCone(U1, [P01, PWts1...])
        wsos_cone2 = HYP.WSOSPolyInterpCone(U2, [P02, PWts2...])
        wsos_cone3 = HYP.WSOSPolyInterpCone(U3, [P03, PWts3...])
        JuMP.@objective(model, Min, sum(quad_weights[u] * w(pts1[u, :]) for u in 1:U1))
        JuMP.@constraints(model, begin
            [-dvdt(pts2[u, :]) for u in 1:U2] in wsos_cone2
            [diffwv(pts1[u, :]) for u in 1:U1] in wsos_cone1
            [vT(pts3[u, :]) for u in 1:U3] in wsos_cone3
            [w(pts1[u, :]) for u in 1:U1] in wsos_cone1
        end)
    else
        int_box_mon(mon) = prod(1 / (p + 1) - (-1)^(p + 1) / (p + 1) for p in DP.exponents(mon))
        int_box(pol) = sum(DP.coefficient(t) * int_box_mon(t) for t in DP.terms(pol))
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@objective(model, Min, int_box(w))
        JuMP.@constraint(model, -dvdt >= 0, domain = (SAS.@set -1 <= x  && x <= 1 && 0 <= t && t <= 1))
        JuMP.@constraint(model, diffwv >= 0, domain = (SAS.@set -1 <= x && x <= 1))
        JuMP.@constraint(model, vT >= 0, domain = (SAS.@set -0.01 <= x && x <= 0.01))
        JuMP.@constraint(model, w >= 0, domain = (SAS.@set -1 <= x && x <= 1))
    end
    return (model = model,)
end

regionofattr1_JuMP() = regionofattr_JuMP(4, use_WSOS = true)
regionofattr2_JuMP() = regionofattr_JuMP(4, use_WSOS = false)

function test_regionofattr_JuMP(instance::Function; options)
    data = instance()
    JuMP.optimize!(data.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(data.model) == MOI.OPTIMAL
    return
end

test_regionofattr_JuMP(; options...) = test_regionofattr_JuMP.([
    regionofattr1_JuMP,
    regionofattr2_JuMP,
    ], options = options)
