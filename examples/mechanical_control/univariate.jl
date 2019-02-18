#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Univariate cubic dynamical system example from "Convex computation of the region of attraction of polynomial control systems" by D. Henrion and M. Korda
=#
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const SO = HYP.Solvers
const MO = HYP.Models
const MU = HYP.ModelUtilities

import JuMP
import SumOfSquares
import MathOptInterface
const MOI = MathOptInterface
import PolyJuMP
import DynamicPolynomials
using LinearAlgebra
import Random
import Distributions
using Test

T = 100.0

function get_cones_WSOS(dom1, dom2, dom3, deg)
    (U2, pts2, P02, PWts2, _) = MU.interpolate(dom2, div(deg, 2) + 1, sample = false)
    (U3, pts3, P03, PWts3, _) = MU.interpolate(dom3, div(deg, 2), sample = false)
    wsos_cone2 = HYP.WSOSPolyInterpCone(U2, [P02, PWts2...])
    wsos_cone3 = HYP.WSOSPolyInterpCone(U3, [P03, PWts3...])
    return (wsos_cone2, wsos_cone3, U2, U3, pts2, pts3)
end

function get_bss(dom1, dom2, dom3, x, t)
    bss1 = MU.get_domain_inequalities(dom1, x)
    bss2 = MU.get_domain_inequalities(dom2, [x; t])
    bss3 = MU.get_domain_inequalities(dom3, x)
    return (bss1, bss2, bss2)
end


function build_univariate(deg::Int; use_wsos::Bool = true)
    DynamicPolynomials.@polyvar x
    DynamicPolynomials.@polyvar t
    f = x * (x - 0.5) * (x + 0.5) * T
    dom1 = MU.Box([-1.0], [1.0]) # just state
    dom2 = MU.Box([-1.0, 0.0], [1.0, 1.0]) # sate and time
    dom3 = MU.Box([-0.01], [0.01]) # state at the end

    (U1, pts1, P01, PWts1, quad_weights) = MU.interpolate(dom1, div(deg, 2) + 1, sample = false, calc_w = true)
    wsos_cone1 = HYP.WSOSPolyInterpCone(U1, [P01, PWts1...])

    model = SumOfSquares.SOSModel(JuMP.with_optimizer(Hypatia.Optimizer, verbose=true))
    JuMP.@variables(model, begin
        v, PolyJuMP.Poly(DynamicPolynomials.monomials([x; t], 0:deg)) # TODO issue reverse order
        w, PolyJuMP.Poly(DynamicPolynomials.monomials(x, 0:deg))
    end)

    JuMP.@objective(model, Min, sum(quad_weights[u] * w(pts1[u, :]) for u in 1:U1))

    delvdelx = DynamicPolynomials.differentiate(v, x)
    delvdelt = DynamicPolynomials.differentiate(v, t)
    dvdt = delvdelt + delvdelx * f
    diffwv = w - DynamicPolynomials.subs(v, t => 0.0) - 1.0
    vT = DynamicPolynomials.subs(v, t => 1.0)

    if use_wsos
        (wsos_cone2, wsos_cone3, U2, U3, pts2, pts3) = get_cones_WSOS(dom1, dom2, dom3, deg)
        JuMP.@constraints(model, begin
            [-dvdt(pts2[u, :]) for u in 1:U2] in wsos_cone2
            [diffwv(pts1[u, :]) for u in 1:U1] in wsos_cone1
            [vT(pts3[u, :]) for u in 1:U3] in wsos_cone3
            [w(pts1[u, :]) for u in 1:U1] in wsos_cone1
        end)
    else
        (bss1, bss2, bss2) = get_bss(dom1, dom2, dom3, x, t)
        # JuMP.@constraint(model, -dvdt in JuMP.PSDCone(), domain = bss2)
        JuMP.@constraint(model, diffwv in JuMP.PSDCone(), domain = bss1)
        JuMP.@constraint(model, vT in JuMP.PSDCone(), domain = bss3)
        JuMP.@constraint(model, w in JuMP.PSDCone(), domain = bss1)
    end

    return model
end

function run_JuMP_univariate_roa()
    model = build_univariate(4, use_wsos = false)
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

    return nothing
end
