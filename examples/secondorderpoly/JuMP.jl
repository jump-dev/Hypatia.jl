#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

check a sufficient condition for pointwise membership of vector valued polynomials in the second order cone
=#

import LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
const DP = DynamicPolynomials
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

const rt2 = sqrt(2)

function secondorderpolyJuMP(polyvec::Vector, deg::Int)
    model = JuMP.Model()
    halfdeg = div(deg + 1, 2)
    (U, pts, P0, _, _) = MU.interpolate(MU.FreeDomain(1), halfdeg, sample = false)
    cone = HYP.WSOSPolyInterpSOCCone(length(polyvec), U, [P0])
    JuMP.@constraint(model, [polyvec[i](pts[u, :]) for i in 1:length(polyvec) for u in 1:U] in cone)
    return (model,)
end

secondorderpolyJuMP1() = (DP.@polyvar x; secondorderpolyJuMP([2x^2 + 2, x, x], 2))
secondorderpolyJuMP2() = (DP.@polyvar x; secondorderpolyJuMP([x^2 + 2, x], 2))
secondorderpolyJuMP3() = (DP.@polyvar x; secondorderpolyJuMP([x^2 + 2, x, x], 2))
secondorderpolyJuMP4() = (DP.@polyvar x; secondorderpolyJuMP([2 * x^4 + 8 * x^2 + 4, x + 2 + (x + 1)^2, x], 4))
secondorderpolyJuMP5() = (DP.@polyvar x; secondorderpolyJuMP([x, x^2 + x], 2))
secondorderpolyJuMP6() = (DP.@polyvar x; secondorderpolyJuMP([x, x + 1], 2))
secondorderpolyJuMP7() = (DP.@polyvar x; secondorderpolyJuMP([x^2, x], 2))
secondorderpolyJuMP8() = (DP.@polyvar x; secondorderpolyJuMP([x + 2, x], 2))
secondorderpolyJuMP9() = (DP.@polyvar x; secondorderpolyJuMP([x - 1, x, x], 2))

function test_secondorderpolyJuMP(instance; options)
    (instance, isfeas) = instance
    (model,) = instance()
    JuMP.optimize!(model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(model) == (isfeas ? MOI.OPTIMAL : MOI.INFEASIBLE)
end

test_secondorderpolyJuMP(; options...) = test_secondorderpolyJuMP.([
    (secondorderpolyJuMP1, true),
    (secondorderpolyJuMP2, true),
    (secondorderpolyJuMP3, true),
    (secondorderpolyJuMP4, true),
    (secondorderpolyJuMP5, false),
    (secondorderpolyJuMP6, false),
    (secondorderpolyJuMP7, false),
    (secondorderpolyJuMP8, false),
    (secondorderpolyJuMP9, false),
    ], options = options)

#
# @testset "everything" begin
#     simple_feasibility()
#     simple_infeasibility()
#
#     Random.seed!(1)
#     for deg in 1:2, n in 1:2, npolys in 1:2
#         println()
#         @show deg, n, npolys
#
#         dom = MU.FreeDomain(n)
#         d = div(deg + 1, 2)
#         (U, pts, P0, _, w) = MU.interpolate(dom, d, sample = false, calc_w = true)
#         lagrange_polys = MU.recover_lagrange_polys(pts, 2d)
#
#         # generate vector of random polys using the Lagrange basis
#         random_coefs = Random.rand(npolys, U)
#         subpolys = [LinearAlgebra.dot(random_coefs[i, :], lagrange_polys) for i in 1:npolys]
#         random_vec = [random_coefs[i, u] for i in 1:npolys for u in 1:U]
#
#         model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, max_iters = 100))
#         JuMP.@variable(model, coefs[1:U])
#         JuMP.@constraint(model, [coefs; random_vec...] in HYP.WSOSPolyInterpSOCCone(npolys + 1, U, [P0]))
#         # JuMP.@objective(model, Min, dot(quad_weights, coefs))
#         JuMP.optimize!(model)
#         upper_bound = LinearAlgebra.dot(JuMP.value.(coefs), lagrange_polys)
#         @test JuMP.termination_status(model) == MOI.OPTIMAL
#         @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
#
#         for i in 1:50
#             pt = randn(n)
#             @test (upper_bound(pt))^2 >= sum(subpolys.^2)(pt)
#         end
#     end
# end
