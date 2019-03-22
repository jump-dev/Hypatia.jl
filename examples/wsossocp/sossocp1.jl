#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
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
using Test
import Random

const rt2 = sqrt(2)

function JuMP_polysoc(P)
    Random.seed!(1)
    dom = MU.FreeDomain(1)
    d = div(maximum(DynamicPolynomials.maxdegree.(P)) + 1, 2)
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)
    mat_wsos_cone = HYP.WSOSPolyInterpSOCCone(2, U, [P0])
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@constraint(model, [P[i](pts[u, :]) for i in 1:2 for u in 1:U] in mat_wsos_cone)
    return model
end

DynamicPolynomials.@polyvar x
for socpoly in [[x + 1; x]]
    model = JuMP_polysoc(socpoly)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
end

for socpoly in [[x; x + 1]]
    model = JuMP_polysoc(socpoly)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.INFEASIBLE
    @test JuMP.primal_status(model) == MOI.INFEASIBLE_POINT
end
