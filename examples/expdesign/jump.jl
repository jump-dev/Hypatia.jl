#=
Copyright 2018, Chris Coey and contributors

D-optimal experimental design
adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.5
  maximize    logdet(V*diagm(np)*V')
  subject to  sum(np) == n
              0 .<= np .<= nmax
where np is a vector of variables representing the number of experiment p to run (fractional),
and the columns of V are the vectors representing each experiment
=#

import Hypatia
import MathOptInterface
const MOI = MathOptInterface
import JuMP
using LinearAlgebra
import Random
using Test

function build_JuMP_expdesign(
    q::Int,
    p::Int,
    V::Matrix{Float64},
    n::Int,
    nmax::Int,
    )
    @assert (p > q) && (n > q) && (nmax <= n)
    @assert size(V) == (q, p)

    model = JuMP.Model(JuMP.with_optimizer(Hypatia.Optimizer, verbose = true))
    JuMP.@variable(model, hypo) # hypograph of logdet variable
    JuMP.@objective(model, Max, hypo)
    JuMP.@variable(model, 0 <= np[1:p] <= nmax) # number of each experiment
    JuMP.@constraint(model, sum(np) == n) # n experiments total
    Q = V * diagm(np) * V' # information matrix
    JuMP.@constraint(model, vcat(hypo, 1.0, [Q[i, j] for i in 1:q for j in 1:i]) in MOI.LogDetConeTriangle(q)) # hypograph of logdet of information matrix

    return (model, np)
end

function JuMP_expdesign1()
    (q, p, n, nmax) = (25, 75, 125, 5) # large
    V = randn(q, p)
    return build_JuMP_expdesign(q, p, V, n, nmax)
end

function JuMP_expdesign2()
    (q, p, n, nmax) = (10, 30, 50, 5) # medium
    V = randn(q, p)
    return build_JuMP_expdesign(q, p, V, n, nmax)
end

function JuMP_expdesign3()
    (q, p, n, nmax) = (5, 15, 25, 5) # small
    V = randn(q, p)
    return build_JuMP_expdesign(q, p, V, n, nmax)
end

function JuMP_expdesign4()
    (q, p, n, nmax) = (4, 8, 12, 3) # tiny
    V = randn(q, p)
    return build_JuMP_expdesign(q, p, V, n, nmax)
end

function JuMP_expdesign5()
    (q, p, n, nmax) = (3, 5, 7, 2) # miniscule
    V = randn(q, p)
    return build_JuMP_expdesign(q, p, V, n, nmax)
end

function run_JuMP_expdesign(; rseed::Int = 1)
    Random.seed!(rseed)
    (model, np) = JuMP_expdesign3()
    JuMP.optimize!(model)

    term_status = JuMP.termination_status(model)
    primal_obj = JuMP.objective_value(model)
    dual_obj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)
    npval = JuMP.value.(np)

    @test term_status == MOI.OPTIMAL
    @test pr_status == MOI.FEASIBLE_POINT
    @test du_status == MOI.FEASIBLE_POINT
    @test primal_obj ≈ dual_obj atol = 1e-4 rtol = 1e-4
    @test primal_obj ≈ logdet(V * Diagonal(npval) * V') atol = 1e-4 rtol = 1e-4
    @test sum(npval) ≈ n atol = 1e-4 rtol = 1e-4
    @test all(-1e-4 .<= npval .<= nmax + 1e-4)

    return
end
