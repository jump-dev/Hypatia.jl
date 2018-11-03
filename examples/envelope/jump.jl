#=
Copyright 2018, Chris Coey and contributors

see description in examples/envelope/native.jl, which formulates the dual envelope problem in the native interface; here we formulate the corresponding primal problem
=#

using Hypatia
import MathOptInterface
MOI = MathOptInterface
using JuMP
# using MultivariatePolynomials
# using DynamicPolynomials
# using SemialgebraicSets
# using PolyJuMP
# using SumOfSquares
using LinearAlgebra
using Random
using Test

function build_JuMP_envelope(
    npoly::Int,
    deg::Int,
    n::Int,
    d::Int;
    rseed::Int = 1,
    )
    # generate interpolation
    # TODO this should be built into the modeling layer
    @assert deg <= d
    (L, U, pts, P0, P, w) = Hypatia.interpolate(n, d, calc_w=true)
    LWts = fill(binomial(n+d-1, n), n)
    wtVals = 1.0 .- pts.^2
    PWts = [Array((qr(Diagonal(sqrt.(wtVals[:, j])) * P[:, 1:LWts[j]])).Q) for j in 1:n]

    # generate random polynomials
    Random.seed!(rseed)
    LDegs = binomial(n+deg, n)
    polys = P0[:, 1:LDegs]*rand(-9:9, LDegs, npoly)

    # build JuMP model
    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, fpv[j in 1:U]) # values at Fekete points
    @objective(model, Max, dot(fpv, w)) # integral over domain (via quadrature)
    @constraint(model, [i in 1:npoly], polys[:,i] .- fpv in WSOSPolyInterpCone(U, [P, PWts...]))

    return (model, fpv)
end

function run_JuMP_envelope()
    (npoly, deg, n, d) =
        2, 3, 1, 4
        # 2, 3, 2, 4
        # 2, 3, 3, 4

    (model, fpv) = build_JuMP_envelope(npoly, deg, n, d)
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
