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
import LinearAlgebra

function run_transformed_sospoly()
    Random.seed!(1)

    DynamicPolynomials.@polyvar x
    poly = (x + 1)^2
    dom = MU.FreeDomain(1)

    d = 3
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)
    transform = Matrix{Float64}(LinearAlgebra.I, U, U) * 5.0 #rand(1, U)
    mono_dual_cone = HYP.MonotonicPolyCone(U, [P0], transform, true)

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@variable(model, y[1:U])
    JuMP.@constraint(model, y in mono_dual_cone)
    JuMP.@objective(model, Min, sum(y[u] * poly(pts[u, :]...) for u in 1:U))

    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    return
end

# function run_JuMP_sosmat1()
#     Random.seed!(1)
#
#     DynamicPolynomials.@polyvar x
#     poly0 = x
#     # poly1 = DynamicPolynomials.differentiate(poly0)
#     dom = MU.FreeDomain(1)
#
#     d = 3
#     (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)
#     mat_wsos_cone = HYP.WSOSPolyInterpMatCone(2, U, [P0])
#     V = [pts[i]^j for i in 1:U, j in 1:U]
#
#     model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
#     JuMP.@constraint(model, [P[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)
#
#     JuMP.optimize!(model)
#     @test JuMP.termination_status(model) == MOI.OPTIMAL
#     @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
#     return
# end
