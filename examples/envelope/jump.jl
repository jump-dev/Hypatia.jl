#=
Copyright 2018, Chris Coey and contributors

see description in examples/envelope/native.jl
=#

using Hypatia
import MathOptInterface
MOI = MathOptInterface
using JuMP
using LinearAlgebra
using Random
using Test

function build_JuMP_envelope_boxinterp(
    npoly::Int,
    deg::Int,
    n::Int,
    d::Int;
    rseed::Int = 1,
    )
    # generate interpolation
    @assert deg <= d
    (L, U, pts, P0, w) = Hypatia.interp_box(n, d, calc_w=true)
    P0sub = view(P0, :, 1:binomial(n+d-1, n))
    Wtsfun = (j -> sqrt.(1.0 .- abs2.(pts[:,j])))
    PWts = [Wtsfun(j) .* P0sub for j in 1:n]

    # generate random polynomials
    Random.seed!(rseed)
    LDegs = binomial(n+deg, n)
    polys = P0[:, 1:LDegs]*rand(-9:9, LDegs, npoly)

    # build JuMP model
    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, fpv[j in 1:U]) # values at Fekete points
    @objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    @constraint(model, [i in 1:npoly], polys[:,i] .- fpv in WSOSPolyInterpCone(U, [P0, PWts...]))

    return (model, fpv)
end

function run_JuMP_envelope_boxinterp()
    (npoly, deg, n, d) =
        # (2, 3, 1, 4)
        # (2, 3, 2, 4)
        (2, 3, 3, 4)

    (model, fpv) = build_JuMP_envelope_boxinterp(npoly, deg, n, d)
    JuMP.optimize!(model)

    term_status = JuMP.termination_status(model)
    pobj = JuMP.objective_value(model)
    dobj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4

    return nothing
end

function build_JuMP_envelope_sampleinterp(
    npoly::Int,
    deg::Int,
    n::Int,
    d::Int,
    domain::Hypatia.InterpDomain;
    rseed::Int = 1,
    )
    # generate interpolation
    @assert deg <= d
    (Ldegs, U, pts, P0, PWts, w) = Hypatia.interp_sample(domain, n, d, calc_w=true)

    # generate random polynomials
    Random.seed!(rseed)
    LDegs = binomial(n+deg, n)
    polys = P0[:, 1:LDegs]*rand(-9:9, LDegs, npoly)

    # build JuMP model
    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, fpv[j in 1:U]) # values at Fekete points
    @objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    @constraint(model, [i in 1:npoly], polys[:,i] .- fpv in WSOSPolyInterpCone(U, [P0, PWts...]))

    return (model, fpv)
end

function run_JuMP_envelope_sampleinterp(dom::Hypatia.InterpDomain)
    n = Hypatia.dimension(dom)
    (npoly, deg, d) = (2, 3, 4)

    (model, fpv) = build_JuMP_envelope_sampleinterp(npoly, deg, n, d, dom)
    JuMP.optimize!(model)

    term_status = JuMP.termination_status(model)
    pobj = JuMP.objective_value(model)
    dobj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4

    return nothing
end

run_JuMP_envelope_sampleinterp_box() = run_JuMP_envelope_sampleinterp(Hypatia.Box(-ones(2), ones(2)))
run_JuMP_envelope_sampleinterp_ball() = run_JuMP_envelope_sampleinterp(Hypatia.Ball(zeros(2), sqrt(2)))
