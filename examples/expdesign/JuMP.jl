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
    use_logdet::Bool = true,
    )
    @assert (p > q) && (n > q) && (nmax <= n)
    V = randn(q, p)

    model = JuMP.Model()
    JuMP.@variable(model, 0 <= np[1:p] <= nmax) # number of each experiment
    Q = V * diagm(np) * V' # information matrix
    JuMP.@constraint(model, sum(np) == n) # n experiments total
    v1 = [Q[i, j] for i in 1:q for j in 1:i] # vectorized Q

    if use_logdet
        JuMP.@variable(model, hypo) # hypograph of logdet variable
        JuMP.@objective(model, Max, hypo)
        JuMP.@constraint(model, vcat(hypo, 1.0, v1) in MOI.LogDetConeTriangle(q)) # hypograph of logdet of information matrix
    else
        JuMP.@variables(model, begin
            hypo[1:q]
            lowertri[i in 1:q, j in 1:i]
        end)
        v2 = vcat([vcat(zeros(i - 1), [lowertri[j, i] for j in i:q], zeros(i - 1), lowertri[i, i]) for i in 1:q]...)
        JuMP.@constraints(model, begin
            vcat(v1, v2) in MOI.PositiveSemidefiniteConeTriangle(2q)
            [i in 1:q], [hypo[i], 1.0, lowertri[i, i]] in MOI.ExponentialCone()
        end)
        JuMP.@objective(model, Max, sum(hypo))
    end
    return (model = model,)
end

expdesignJuMP1() = expdesignJuMP(25, 75, 125, 5)
expdesignJuMP2() = expdesignJuMP(10, 30, 50, 5)
expdesignJuMP3() = expdesignJuMP(5, 15, 25, 5)
expdesignJuMP4() = expdesignJuMP(4, 8, 12, 3)
expdesignJuMP5() = expdesignJuMP(3, 5, 7, 2)
expdesignJuMP6() = expdesignJuMP(25, 75, 125, 5, use_logdet = false)
expdesignJuMP7() = expdesignJuMP(10, 30, 50, 5, use_logdet = false)
expdesignJuMP8() = expdesignJuMP(5, 15, 25, 5, use_logdet = false)
expdesignJuMP9() = expdesignJuMP(4, 8, 12, 3, use_logdet = false)
expdesignJuMP10() = expdesignJuMP(3, 5, 7, 2, use_logdet = false)

expdesignJuMP11() = expdesignJuMP(50, 100, 125, 5, use_logdet = true)
expdesignJuMP12() = expdesignJuMP(50, 100, 125, 5, use_logdet = false)

function test_expdesignJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    @time JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_expdesignJuMP_all(; options...) = test_expdesignJuMP.([
    expdesignJuMP1,
    expdesignJuMP2,
    expdesignJuMP3,
    expdesignJuMP4,
    expdesignJuMP5,
    expdesignJuMP6,
    expdesignJuMP7,
    expdesignJuMP9,
    expdesignJuMP9,
    expdesignJuMP10,
    ], options = options)

test_expdesignJuMP(; options...) = test_expdesignJuMP.([
    expdesignJuMP1,
    expdesignJuMP6,
    # expdesignJuMP3,
    # expdesignJuMP8,
    ], options = options)
