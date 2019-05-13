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

function expdesign_JuMP(
    q::Int,
    p::Int,
    n::Int,
    nmax::Int;
    rseed::Int = 1,
    )
    Random.seed!(rseed)
    @assert (p > q) && (n > q) && (nmax <= n)
    V = randn(q, p)
    model = JuMP.Model()
    JuMP.@variable(model, 0 <= np[1:p] <= nmax) # number of each experiment
    JuMP.@variable(model, hypo) # hypograph of logdet variable
    JuMP.@objective(model, Max, hypo)
    JuMP.@constraint(model, sum(np) == n) # n experiments total
    Q = V * diagm(np) * V' # information matrix
    JuMP.@constraint(model, vcat(hypo, 1.0, [Q[i, j] for i in 1:q for j in 1:i]) in MOI.LogDetConeTriangle(q)) # hypograph of logdet of information matrix
    return (model = model, n = n, nmax = nmax, V = V, np = np)
end

expdesign1_JuMP() = expdesign_JuMP(25, 75, 125, 5) # large
expdesign2_JuMP() = expdesign_JuMP(10, 30, 50, 5) # medium
expdesign3_JuMP() = expdesign_JuMP(5, 15, 25, 5) # small
expdesign4_JuMP() = expdesign_JuMP(4, 8, 12, 3) # tiny
expdesign5_JuMP() = expdesign_JuMP(3, 5, 7, 2) # miniscule

function test_expdesign_JuMP(builder::Function; options)
    data = builder()
    JuMP.optimize!(data.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(data.model) == MOI.OPTIMAL
    npval = JuMP.value.(data.np)
    @test JuMP.objective_value(data.model) ≈ logdet(data.V * Diagonal(npval) * data.V') atol = 1e-4 rtol = 1e-4
    @test sum(npval) ≈ data.n atol = 1e-4 rtol = 1e-4
    @test all(-1e-4 .<= npval .<= data.nmax + 1e-4)
    return
end

test_expdesign_JuMP(; options...) = test_expdesign_JuMP.([
    # expdesign1_JuMP,
    # expdesign2_JuMP,
    expdesign3_JuMP,
    expdesign4_JuMP,
    expdesign5_JuMP,
    ], options = options)

test_expdesign_JuMP_small(; options...) = test_expdesign_JuMP.([expdesign3_JuMP], options = options)
