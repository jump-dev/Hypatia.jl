#=
Copyright 2018, Chris Coey and contributors

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
using LinearAlgebra
import Random
using Test

function build_quadrature(
    d::Int,
    p, # polynomial parameter (in objective of moment problem; must be positive on K)
    K_dom::MU.Domain,
    B_dom::MU.Domain; # canonical set (for which we have quadrature weights) containing K_dom
    )
    # generate interpolation for K
    (U, pts, P0, PWts, _) = MU.interpolate(K_dom, d, sample = true, calc_w = false)

    # get quadrature weights for B
    y2 = MU.get_weights(B_dom, pts)

    # build JuMP model
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@variable(model, y1[1:U]) # moments of μ1
    JuMP.@objective(model, Max, sum(y1[i] * p(pts[i, :]) for i in 1:U))
    JuMP.@constraint(model, y1 in HYP.WSOSPolyInterpCone(U, [P0, PWts...], true))
    JuMP.@constraint(model, y2 - y1 in HYP.WSOSPolyInterpCone(U, [P0], true))

    return (model, y1, pts)
end

 
