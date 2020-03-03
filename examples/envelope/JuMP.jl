#=
Copyright 2018, Chris Coey and contributors

see description in examples/envelope/native.jl
=#

using LinearAlgebra
import Random
using Test
import JuMP
const MOI = JuMP.MOI
import Hypatia
const MU = Hypatia.ModelUtilities

function envelope_JuMP(
    T::Type{Float64}, # TODO support generic reals
    n::Int,
    rand_halfdeg::Int,
    num_polys::Int,
    env_halfdeg::Int;
    domain::MU.Domain = MU.Box{T}(-ones(T, n), ones(T, n)),
    sample::Bool = true,
    sample_factor::Int = 100,
    )
    @assert n == MU.get_dimension(domain)
    @assert rand_halfdeg <= env_halfdeg

    # generate interpolation
    (U, pts, Ps, w) = MU.interpolate(domain, env_halfdeg, calc_w = true, sample = sample, sample_factor = sample_factor)

    # generate random polynomials
    L = binomial(n + rand_halfdeg, n)
    polys = Ps[1][:, 1:L] * rand(-9:9, L, num_polys)

    model = JuMP.Model()
    JuMP.@variable(model, fpv[j in 1:U]) # values at Fekete points
    JuMP.@objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    JuMP.@constraint(model, [i in 1:num_polys], polys[:, i] .- fpv in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))

    return (model = model,)
end

function test_envelope_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = envelope_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

envelope_JuMP_fast = [
    (2, 2, 3, 4),
    (2, 3, 2, 4),
    (3, 3, 3, 3),
    (3, 3, 5, 4),
    (5, 2, 5, 2),
    (1, 30, 2, 30),
    (10, 1, 3, 1),
    ]
envelope_JuMP_slow = [
    # TODO
    ]
