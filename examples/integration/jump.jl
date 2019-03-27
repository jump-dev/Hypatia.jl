#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

approximate integration (upper bound) of a polynomial over a basic semialgebraic set
adapted from "Approximate volume and integration for basic semialgebraic sets"
by Henrion, Lasserre, & Savorgnan (2009)
https://pdfs.semanticscholar.org/893b/e70a990901d7b6b2f052cbcb5883a043b5d9.pdf
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
import SemialgebraicSets
# import Random
using Test

function build_quadrature(
    d::Int,
    K_dom::MU.Domain,
    B_dom::MU.Domain,
    ppar,
    )
    # generate interpolation for B
    (U, pts, P0, PWts, w) = MU.interpolate(B_dom, d, sample = true, calc_w = true)

    # get weights for B
    y2 = # TODO

    # build JuMP model
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@variable(model, y[1:U]) # moments of μ1
    JuMP.@objective(model, Max, sum(y[i] * ppar(pts[i, :]) for i in 1:U))
    JuMP.@constraint(model, y in HYP.WSOSPolyInterpCone(U, [P0, PWts...], true))
    JuMP.@constraint(model, w - y in HYP.WSOSPolyInterpCone(U, [P0], true))

    return (model, y, pts)
end

function integrate_poly(
    p, # poly to integrate
    d::Int, # degree of moment problem
    K_dom::MU.Domain, # domain on which quadrature weights are valid
    B_dom::MU.Domain, # canonical set (for which we have quadrature weights) containing K_dom
    ppar, # polynomial as the parameter in objective of moment problem
    )
    # optimize to get quadrature weights
    (model, y, pts) = build_quadrature(d, K_dom, B_dom, ppar)
    JuMP.optimize!(model)

    w = JuMP.value.(y)
    integral = sum(w[i] * p(pts[i, :]) for i in eachindex(w))

    println(w)
    println(integral)
end


n = 1
DynamicPolynomials.@polyvar x
d = 10
p = 1.0 + 0.0x
K_dom = MU.Box([0.0], [0.5]) # TODO SemialgebraicSets.@set(x * (1/2 - x) >= 0)
B_dom = MU.Box([-1.0], [1.0])
ppar = 1.0 + 0.0x

integrate_poly(p, d, K_dom, B_dom, ppar)
