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
    use_psd::Bool = false,
    criterion = "d",
    )
    @assert (p > q) && (n > q) && (nmax <= n)
    @assert criterion in ["d", "e"]
    V = randn(q, p)

    model = JuMP.Model()
    JuMP.@variable(model, 0 <= np[1:p] <= nmax) # number of each experiment
    Q = V * diagm(np) * V' # information matrix
    JuMP.@constraint(model, sum(np) == n) # n experiments total
    v1 = [Q[i, j] for i in 1:q for j in 1:i] # vectorized Q

    if !use_psd
        JuMP.@variable(model, hypo) # hypograph of logdet variable
        JuMP.@objective(model, Max, hypo)
        if criterion == "d"
            JuMP.@constraint(model, vcat(hypo, 1.0, v1) in MOI.LogDetConeTriangle(q)) # hypograph of logdet of information matrix
        else
            error("root det cone not implemented yet")
        end
    else
        JuMP.@variable(model, lowertri[i in 1:q, j in 1:i])
        v2 = vcat([vcat(zeros(i - 1), [lowertri[j, i] for j in i:q], zeros(i - 1), lowertri[i, i]) for i in 1:q]...)
        JuMP.@constraint(model, vcat(v1, v2) in MOI.PositiveSemidefiniteConeTriangle(2q))
        if criterion == "d"
            JuMP.@variable(model, hypo[1:q])
            JuMP.@constraint(model, [i in 1:q], [hypo[i], 1.0, lowertri[i, i]] in MOI.ExponentialCone())
        else
            JuMP.@variable(model, hypo)
            JuMP.@constraint(model, vcat(hypo, [lowertri[i, i] for i in 1:q]) in MOI.GeometricMeanCone(q + 1))
        end
        JuMP.@objective(model, Max, sum(hypo))
    end
    return (model = model,)
end

expdesignJuMP1() = expdesignJuMP(25, 75, 125, 5) # large
expdesignJuMP2() = expdesignJuMP(10, 30, 50, 5) # medium
expdesignJuMP3() = expdesignJuMP(5, 15, 25, 5) # small
expdesignJuMP4() = expdesignJuMP(4, 8, 12, 3) # tiny
expdesignJuMP5() = expdesignJuMP(3, 5, 7, 2) # miniscule
expdesignJuMP6() = expdesignJuMP(25, 75, 125, 5, use_psd = true) # large
expdesignJuMP7() = expdesignJuMP(10, 30, 50, 5, use_psd = true) # medium
expdesignJuMP8() = expdesignJuMP(5, 15, 25, 5, use_psd = true) # small
expdesignJuMP9() = expdesignJuMP(4, 8, 12, 3, use_psd = true) # tiny
expdesignJuMP10() = expdesignJuMP(3, 5, 7, 2, use_psd = true) # miniscule
expdesignJuMP11() = expdesignJuMP(25, 75, 125, 5, use_psd = true, criterion = "e") # large
expdesignJuMP12() = expdesignJuMP(10, 30, 50, 5, use_psd = true, criterion = "e") # medium
expdesignJuMP13() = expdesignJuMP(5, 15, 25, 5, use_psd = true, criterion = "e") # small
expdesignJuMP14() = expdesignJuMP(4, 8, 12, 3, use_psd = true, criterion = "e") # tiny
expdesignJuMP15() = expdesignJuMP(3, 5, 7, 2, use_psd = true, criterion = "e") # miniscule

function test_expdesignJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
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
    expdesignJuMP11,
    expdesignJuMP12,
    expdesignJuMP13,
    expdesignJuMP14,
    expdesignJuMP15,
    ], options = options)

test_expdesignJuMP(; options...) = test_expdesignJuMP.([
    expdesignJuMP3,
    expdesignJuMP8,
    expdesignJuMP13,
    ], options = options)
