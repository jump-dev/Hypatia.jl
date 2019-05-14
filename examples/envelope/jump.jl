#=
Copyright 2018, Chris Coey and contributors

see description in examples/envelope/native.jl
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MU = HYP.ModelUtilities

import MathOptInterface
const MOI = MathOptInterface
import JuMP
using LinearAlgebra
import Random
using Test

function envelope_JuMP(
    npoly::Int,
    deg::Int,
    d::Int,
    domain::MU.Domain;
    sample::Bool = true,
    rseed::Int = 1,
    )
    Random.seed!(rseed)
    # generate interpolation
    @assert deg <= d
    model = JuMP.Model()
    (U, pts, P0, PWts, w) = MU.interpolate(domain, d, sample = sample, calc_w = true)
    # generate random polynomials
    n = MU.get_dimension(domain)
    LDegs = binomial(n + deg, n)
    polys = P0[:, 1:LDegs] * rand(-9:9, LDegs, npoly)
    JuMP.@variable(model, fpv[j in 1:U]) # values at Fekete points
    JuMP.@objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    JuMP.@constraint(model, [i in 1:npoly], polys[:, i] .- fpv in HYP.WSOSPolyInterpCone(U, [P0, PWts...]))
    return (model = model,)
end

envelope1_JuMP() = envelope_JuMP(2, 3, 4, MU.Box(-ones(2), ones(2)))
envelope2_JuMP() = envelope_JuMP(2, 3, 4, MU.Ball(zeros(2), sqrt(2))) # needs fix to work https://github.com/chriscoey/Hypatia.jl/issues/173
envelope3_JuMP() = envelope_JuMP(2, 3, 4, MU.Box(-ones(2), ones(2)), sample = false)

function test_envelope_JuMP(instance::Function; options)
    data = instance()
    JuMP.optimize!(data.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(data.model) == MOI.OPTIMAL
    return
end

test_envelope_JuMP(; options...) = test_envelope_JuMP.([
    envelope1_JuMP,
    envelope2_JuMP,
    envelope3_JuMP,
    ], options = options)
