#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

check a sufficient condition for pointwise membership of vector valued polynomials in the second order cone
=#

import LinearAlgebra
import Random
using Test
import JuMP
const MOI = JuMP.MOI
import Hypatia
const MU = Hypatia.ModelUtilities

function secondorderpoly_JuMP(
    T::Type{Float64}, # TODO support generic reals
    polyvec::Function,
    deg::Int,
    is_feas::Bool, # whether model should be primal-dual feasible; only for testing
    )
    halfdeg = div(deg + 1, 2)
    (U, pts, Ps, _) = MU.interpolate(MU.FreeDomain{Float64}(1), halfdeg, sample = false)
    vals = polyvec.(pts)
    l = length(vals[1])
    cone = Hypatia.WSOSInterpEpiNormEuclCone{Float64}(l, U, Ps)

    model = JuMP.Model()
    JuMP.@constraint(model, [v[i] for i in 1:l for v in vals] in cone)

    return (model = model, is_feas = is_feas)
end

function test_secondorderpoly_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = secondorderpoly_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == (d.is_feas ? MOI.OPTIMAL : MOI.INFEASIBLE)
    return d.model.moi_backend.optimizer.model.optimizer.result
end

secondorderpoly_JuMP_fast = [
    (x -> [2x^2 + 2, x, x], 2, true),
    (x -> [x^2 + 2, x], 2, true),
    (x -> [x^2 + 2, x, x], 2, true),
    (x -> [2 * x^4 + 8 * x^2 + 4, x + 2 + (x + 1)^2, x], 4, true),
    (x -> [x, x^2 + x], 2, false),
    (x -> [x, x + 1], 2, false),
    (x -> [x^2, x], 2, false),
    (x -> [x + 2, x], 2, false),
    (x -> [x - 1, x, x], 2, false),
    ]
secondorderpoly_JuMP_slow = [
    # TODO
    ]
