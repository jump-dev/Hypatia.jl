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
    return (model = model,)
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
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == (isfeas ? MOI.OPTIMAL : MOI.INFEASIBLE)
end

test_secondorderpolyJuMP_all(; options...) = test_secondorderpolyJuMP.([
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

test_secondorderpolyJuMP(; options...) = test_secondorderpolyJuMP.([
    (secondorderpolyJuMP1, true),
    (secondorderpolyJuMP6, false),
    ], options = options)
