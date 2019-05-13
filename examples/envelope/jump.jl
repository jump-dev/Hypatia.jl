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

function build_envelope_JuMP(
    model::JuMP.Model,
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
    (U, pts, P0, PWts, w) = MU.interpolate(domain, d, sample = sample, calc_w = true)

    # generate random polynomials
    n = MU.get_dimension(domain)
    LDegs = binomial(n + deg, n)
    polys = P0[:, 1:LDegs] * rand(-9:9, LDegs, npoly)

    JuMP.@variable(model, fpv[j in 1:U]) # values at Fekete points
    JuMP.@objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    JuMP.@constraint(model, [i in 1:npoly], polys[:, i] .- fpv in HYP.WSOSPolyInterpCone(U, [P0, PWts...]))

    return model
end

function envelope_JuMP(npoly::Int, deg::Int, d::Int, domain::MU.Domain; sample::Bool = true)
    model = JuMP.Model()
    return build_envelope_JuMP(model, npoly, deg, d, domain, sample = sample)
end

envelope1_JuMP() = envelope_JuMP(2, 3, 4, MU.Box(-ones(2), ones(2)))
envelope2_JuMP() = envelope_JuMP(2, 3, 4, MU.Ball(zeros(2), sqrt(2))) # needs fix to work https://github.com/chriscoey/Hypatia.jl/issues/173
envelope3_JuMP() = envelope_JuMP(2, 3, 4, MU.Box(-ones(2), ones(2)), sample = false)

function test_envelope_JuMP(instance::Function; options)
    model = instance()
    JuMP.optimize!(model, JuMP.with_optimizer(Hypatia.Optimizer; options...))

    term_status = JuMP.termination_status(model)
    primal_obj = JuMP.objective_value(model)
    dual_obj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.OPTIMAL
    @test pr_status == MOI.FEASIBLE_POINT
    @test du_status == MOI.FEASIBLE_POINT
    @test primal_obj ≈ dual_obj atol = 1e-4 rtol = 1e-4

    return
end

test_envelope_JuMP(; options...) = test_envelope_JuMP.([
    envelope1_JuMP,
    envelope2_JuMP,
    envelope3_JuMP,
    ], options = options)
