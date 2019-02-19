#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/sosdemo9.jl
Section 3.9 of SOSTOOLS User's Manual, see https://www.cds.caltech.edu/sostools/
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

function run_JuMP_sosmat3(use_dual::Bool)
    Random.seed!(1)

    DynamicPolynomials.@polyvar x1 x2 x3
    P = [
        (x1^4 + x1^2 * x2^2 + x1^2 * x3^2) (x1 * x2 * x3^2 - x1^3 * x2 - x1 * x2 * (x2^2 + 2 * x3^2));
        (x1 * x2 * x3^2 - x1^3 * x2 - x1 * x2 * (x2^2 + 2 * x3^2)) (x1^2 * x2^2 + x2^2 * x3^2 + (x2^2 + 2 * x3^2)^2);
        ]
    dom = MU.FreeDomain(3)

    d = div(maximum(DynamicPolynomials.maxdegree.(P)) + 1, 2)
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample = false)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    mat_wsos_cone = HYP.WSOSPolyInterpMatCone(2, U, [P0], use_dual)

    if use_dual
        JuMP.@variable(model, z[i in 1:2, 1:i, 1:U])
        JuMP.@constraint(model, [z[i, j, u] * (i == j ? 1.0 : rt2) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)
        JuMP.@objective(model, Min, sum(z[i, j, u] * P[i, j](pts[u, :]...) * (i == j ? 1.0 : 2.0) for i in 1:2 for j in 1:i for u in 1:U))
    else
        JuMP.@constraint(model, [P[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)
    end
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    return nothing
end

run_JuMP_sosmat3_primal() = run_JuMP_sosmat3(false)
run_JuMP_sosmat3_dual() = run_JuMP_sosmat3(true)
