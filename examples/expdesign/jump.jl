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

function build_expdesign_JuMP(
    model::JuMP.Model,
    np::Vector{JuMP.VariableRef},
    q::Int,
    p::Int,
    V::Matrix{Float64},
    n::Int,
    nmax::Int,
    )
    @assert (p > q) && (n > q) && (nmax <= n)
    @assert size(V) == (q, p)

    JuMP.@variable(model, hypo) # hypograph of logdet variable
    JuMP.@objective(model, Max, hypo)
    JuMP.@constraint(model, sum(np) == n) # n experiments total
    Q = V * diagm(np) * V' # information matrix
    JuMP.@constraint(model, vcat(hypo, 1.0, [Q[i, j] for i in 1:q for j in 1:i]) in MOI.LogDetConeTriangle(q)) # hypograph of logdet of information matrix

    return model
end

function expdesign_JuMP(q::Int, p::Int, n::Int, nmax::Int; use_dense::Bool = false, rseed::Int = 1)
    Random.seed!(rseed)
    model = JuMP.Model(JuMP.with_optimizer(Hypatia.Optimizer, verbose = true, use_dense = use_dense))
    JuMP.@variable(model, 0 <= np[1:p] <= nmax) # number of each experiment
    V = randn(q, p)
    build_expdesign_JuMP(model, np, q, p, V, n, nmax)
    return (model = model, n = n, nmax = nmax, V = V, np = np)
end

expdesign1_JuMP(; use_dense::Bool = false) = expdesign_JuMP(25, 75, 125, 5, use_dense = use_dense) # large
expdesign2_JuMP(; use_dense::Bool = false) = expdesign_JuMP(10, 30, 50, 5, use_dense = use_dense) # medium
expdesign3_JuMP(; use_dense::Bool = false) = expdesign_JuMP(5, 15, 25, 5, use_dense = use_dense) # small
expdesign4_JuMP(; use_dense::Bool = false) = expdesign_JuMP(4, 8, 12, 3, use_dense = use_dense) # tiny
expdesign5_JuMP(; use_dense::Bool = false) = expdesign_JuMP(3, 5, 7, 2, use_dense = use_dense) # miniscule


function test_expdesign_JuMP(instance::Function)
    (model, n, nmax, V, np) = instance()
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

test_expdesign_JuMP_many() = test_expdesign_JuMP.([
    # expdesign1_JuMP,
    # expdesign2_JuMP,
    expdesign3_JuMP,
    expdesign4_JuMP,
    expdesign5_JuMP,
])

test_expdesign_JuMP_small() = test_expdesign_JuMP.([expdesign3_JuMP])
