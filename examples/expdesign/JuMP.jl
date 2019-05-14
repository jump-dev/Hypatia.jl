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

using LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia

function expdesignJuMP(
    q::Int,
    p::Int,
    n::Int,
    nmax::Int;
    )
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

expdesignJuMP1() = expdesignJuMP(25, 75, 125, 5) # large
expdesignJuMP2() = expdesignJuMP(10, 30, 50, 5) # medium
expdesignJuMP3() = expdesignJuMP(5, 15, 25, 5) # small
expdesignJuMP4() = expdesignJuMP(4, 8, 12, 3) # tiny
expdesignJuMP5() = expdesignJuMP(3, 5, 7, 2) # miniscule

function test_expdesignJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    npval = JuMP.value.(d.np)
    @test JuMP.objective_value(d.model) ≈ logdet(Symmetric(d.V * Diagonal(npval) * d.V')) atol = 1e-4 rtol = 1e-4
    @test sum(npval) ≈ d.n atol = 1e-4 rtol = 1e-4
    @test all(-1e-4 .<= npval .<= d.nmax + 1e-4)
    return
end

test_expdesignJuMP(; options...) = test_expdesignJuMP.([
    # expdesignJuMP1,
    # expdesignJuMP2,
    expdesignJuMP3,
    expdesignJuMP4,
    expdesignJuMP5,
    ], options = options)

test_expdesignJuMP_quick(; options...) = test_expdesignJuMP.([
    expdesignJuMP3,
    ], options = options)
