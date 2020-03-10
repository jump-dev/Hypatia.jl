#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find a polynomial f such that f² >= Σᵢ gᵢ² where gᵢ are arbitrary polynomials, and the volume under f is minimized

TODO add scalar SOS formulation
=#

using LinearAlgebra
import Random
using Test
import JuMP
const MOI = JuMP.MOI
import Hypatia
const MU = Hypatia.ModelUtilities

function polynorm_JuMP(
    ::Type{T},
    n::Int,
    deg::Int,
    npolys::Int,
    ) where {T <: Float64} # TODO support generic reals
    dom = MU.FreeDomain{Float64}(n)
    halfdeg = div(deg + 1, 2)
    (U, pts, Ps, w) = MU.interpolate(dom, halfdeg, calc_w = true)
    lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)

    model = JuMP.Model()
    JuMP.@variable(model, f[1:U])
    JuMP.@objective(model, Min, dot(w, f))

    L = size(Ps[1], 2)
    polys = Ps[1] * rand(-9:9, L, npolys)

    fpoly = dot(f, lagrange_polys)
    rand_polys = [dot(polys[:, i], lagrange_polys) for i in 1:npolys]
    cone = Hypatia.WSOSInterpEpiNormEuclCone{Float64}(npolys + 1, U, Ps)
    JuMP.@constraint(model, vcat(f, [polys[:, i] for i in 1:npolys]...) in cone)

    return (model, ())
end

function test_polynorm_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = polynorm_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

polynorm_JuMP_fast = [
    (2, 2, 2),
    (2, 1, 3),
    ]
polynorm_JuMP_slow = [
    # TODO
    ]
