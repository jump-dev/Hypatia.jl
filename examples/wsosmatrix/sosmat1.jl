#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/simplematrixsos.jl
Example 3.77 and 3.79 of Blekherman, G., Parrilo, P. A., & Thomas, R. R. (Eds.),
Semidefinite optimization and convex algebraic geometry SIAM 2013
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
# const LS = HYP.LinearSystems
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

function run_JuMP_sosmat1()
    Random.seed!(1)

    DynamicPolynomials.@polyvar x
    P = [(x^2 - 2x + 2) x; x x^2]
    dom = MU.FreeDomain(1)

    d = div(maximum(DynamicPolynomials.maxdegree.(P)) + 1, 2)
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)
    mat_wsos_cone = HYP.WSOSPolyInterpMatCone(2, U, [P0])

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@constraint(model, [P[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)

    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    return
end
