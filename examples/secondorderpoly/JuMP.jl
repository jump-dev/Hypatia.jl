#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

check a sufficient condition for pointwise membership of vector valued polynomials in the second order cone
=#

import Random
using Test
import JuMP
const MOI = JuMP.MOI
import Hypatia
const MU = Hypatia.ModelUtilities

function secondorderpoly_JuMP(
    ::Type{T},
    poly_vec::Function,
    deg::Int,
    is_feas::Bool, # whether model should be primal-dual feasible; only for testing
    ) where {T <: Float64} # TODO support generic reals
    halfdeg = div(deg + 1, 2)
    (U, pts, Ps, _) = MU.interpolate(MU.FreeDomain{Float64}(1), halfdeg, sample = false)
    vals = poly_vec.(pts)
    l = length(vals[1])
    cone = Hypatia.WSOSInterpEpiNormEuclCone{Float64}(l, U, Ps)

    model = JuMP.Model()
    JuMP.@constraint(model, [v[i] for i in 1:l for v in vals] in cone)

    return (model = model, is_feas = is_feas)
end

secondorderpoly_JuMP(
    ::Type{T},
    polys_name::Symbol, 
    args...) where {T <: Float64} = secondorderpoly_JuMP(T, secondorderpoly_data[polys_name], args...)

function test_secondorderpoly_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = secondorderpoly_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == (d.is_feas ? MOI.OPTIMAL : MOI.INFEASIBLE)
    return d.model.moi_backend.optimizer.model.optimizer.result
end

secondorderpoly_data = Dict(
    :polys1 => (x -> [2x^2 + 2, x, x]),
    :polys2 => (x -> [x^2 + 2, x]),
    :polys3 => (x -> [x^2 + 2, x, x]),
    :polys4 => (x -> [2 * x^4 + 8 * x^2 + 4, x + 2 + (x + 1)^2, x]),
    :polys5 => (x -> [x, x^2 + x]),
    :polys6 => (x -> [x, x + 1]),
    :polys7 => (x -> [x^2, x]),
    :polys8 => (x -> [x + 2, x]),
    :polys9 => (x -> [x - 1, x, x]),
    )

secondorderpoly_JuMP_fast = [
    (:polys1, 2, true),
    (:polys2, 2, true),
    (:polys3, 2, true),
    (:polys4, 4, true),
    (:polys5, 2, false),
    (:polys6, 2, false),
    (:polys7, 2, false),
    (:polys8, 2, false),
    (:polys9, 2, false),
    ]
secondorderpoly_JuMP_slow = [
    # TODO
    ]
