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

function build_JuMP_envelope(
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

    # build JuMP model
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@variable(model, fpv[j in 1:U]) # values at Fekete points
    JuMP.@objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    JuMP.@constraint(model, [i in 1:npoly], polys[:, i] .- fpv in HYP.WSOSPolyInterpCone(U, [P0, PWts...]))

    return (model, fpv)
end

function run_JuMP_envelope(
    npoly::Int,
    deg::Int,
    d::Int,
    dom::MU.Domain;
    sample::Bool = true,
    )
    (model, fpv) = build_JuMP_envelope(npoly, deg, d, dom, sample = sample)
    JuMP.optimize!(model)

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

run_JuMP_envelope_sampleinterp_free() = run_JuMP_envelope(2, 1, 4, MU.FreeDomain(2))
run_JuMP_envelope_sampleinterp_box() = run_JuMP_envelope(2, 3, 4, MU.Box(-ones(2), ones(2)))
run_JuMP_envelope_sampleinterp_ball() = run_JuMP_envelope(2, 3, 4, MU.Ball(zeros(2), sqrt(2)))
run_JuMP_envelope_boxinterp() = run_JuMP_envelope(2, 3, 4, MU.Box(-ones(2), ones(2)), sample = false)
