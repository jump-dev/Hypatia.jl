#=
Copyright 2018, Chris Coey and contributors

D-optimal experimental design
adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.5
  maximize    F(V*diagm(np)*V')
  subject to  sum(np) == n
              0 .<= np .<= nmax
where np is a vector of variables representing the number of experiment p to run (fractional),
and the columns of V are the vectors representing each experiment

if logdet_obj or rootdet_obj is true, F is the logdet or rootdet function
if geomean_obj is true, we use a formulation from https://picos-api.gitlab.io/picos/optdes.html that finds an equivalent minimizer
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
    logdet_obj::Bool = false, # use formulation with logdet objective
    rootdet_obj::Bool = false, # use formulation with rootdet objective
    geomean_obj::Bool = false, # use formulation with geomean objective
    use_nat::Bool = true, # use natural for all cones, else extended
    )
    @assert (p > q) && (n > q) && (nmax <= n)
    @assert logdet_obj + geomean_obj + rootdet_obj == 1
    V = randn(q, p)

    model = JuMP.Model()
    JuMP.@variable(model, np[1:p])
    if use_nat
        JuMP.@constraint(model, vcat(nmax / 2, np .- nmax / 2) in MOI.NormInfinityCone(p + 1))
    else
        JuMP.@constraint(model, 0 .<= np)
        JuMP.@constraint(model, np .<= nmax)
    end
    Q = V * diagm(np) * V' # information matrix
    JuMP.@constraint(model, sum(np) == n)
    v1 = [Q[i, j] for i in 1:q for j in 1:i] # vectorized Q

    if (logdet_obj && use_nat) || rootdet_obj || geomean_obj
        # hypograph of logdet/rootdet/geomean
        JuMP.@variable(model, hypo)
        JuMP.@objective(model, Max, hypo)
    end

    if geomean_obj || !use_nat
        JuMP.@variable(model, lowertri[i in 1:q, j in 1:i])
    end

    if !geomean_obj && !use_nat
        v2 = vcat([vcat(zeros(i - 1), [lowertri[j, i] for j in i:q], zeros(i - 1), lowertri[i, i]) for i in 1:q]...)
        JuMP.@constraint(model, vcat(v1, v2) in MOI.PositiveSemidefiniteConeTriangle(2q))
    end

    if logdet_obj
        if use_nat
            JuMP.@constraint(model, vcat(hypo, 1.0, v1) in MOI.LogDetConeTriangle(q)) # hypograph of logdet of information matrix
        else
            JuMP.@variable(model, hypo[1:q])
            JuMP.@constraint(model, [i in 1:q], [hypo[i], 1.0, lowertri[i, i]] in MOI.ExponentialCone())
            JuMP.@objective(model, Max, sum(hypo))
        end
    elseif rootdet_obj
        if use_nat
            JuMP.@constraint(model, vcat(hypo, v1) in MOI.RootDetConeTriangle(q))
        else
            JuMP.@constraint(model, vcat(hypo, [lowertri[i, i] for i in 1:q]) in MOI.GeometricMeanCone(q + 1))
        end
    else
        # hypogeomean/soc formulation
        JuMP.@variable(model, W[1:p, 1:q])
        VW = V * W
        # TODO iterate in constraint
        for i in 1:q
            for j in 1:i
                JuMP.@constraint(model, VW[i, j] == lowertri[i, j])
            end
            for j in (i + 1):q
                JuMP.@constraint(model, VW[i, j] == 0)
            end
        end
        JuMP.@constraints(model, begin
            vcat(hypo, [lowertri[i, i] for i in 1:q]) in MOI.GeometricMeanCone(q + 1)
            [i in 1:p], vcat(sqrt(q) * np[i], W[i, :]) in JuMP.SecondOrderCone()
        end)
    end # obj

    return (model = model,)
end

expdesignJuMP1() = expdesignJuMP(25, 75, 125, 5, logdet_obj = true)
expdesignJuMP2() = expdesignJuMP(10, 30, 50, 5, logdet_obj = true)
expdesignJuMP3() = expdesignJuMP(5, 15, 25, 5, logdet_obj = true)
expdesignJuMP4() = expdesignJuMP(4, 8, 12, 3, logdet_obj = true)
expdesignJuMP5() = expdesignJuMP(3, 5, 7, 2, logdet_obj = true)
expdesignJuMP6() = expdesignJuMP(25, 75, 125, 5, use_nat = false, logdet_obj = true)
expdesignJuMP7() = expdesignJuMP(10, 30, 50, 5, use_nat = false, logdet_obj = true)
expdesignJuMP8() = expdesignJuMP(5, 15, 25, 5, use_nat = false, logdet_obj = true)
expdesignJuMP9() = expdesignJuMP(4, 8, 12, 3, use_nat = false, logdet_obj = true)
expdesignJuMP10() = expdesignJuMP(3, 5, 7, 2, use_nat = false, logdet_obj = true)
expdesignJuMP11() = expdesignJuMP(25, 75, 125, 5, rootdet_obj = true, use_epinorminf = false)
expdesignJuMP12() = expdesignJuMP(10, 30, 50, 5, rootdet_obj = true, use_epinorminf = false)
expdesignJuMP13() = expdesignJuMP(5, 15, 25, 5, rootdet_obj = true, use_epinorminf = false)
expdesignJuMP14() = expdesignJuMP(4, 8, 12, 3, rootdet_obj = true, use_epinorminf = false)
expdesignJuMP15() = expdesignJuMP(3, 5, 7, 2, rootdet_obj = true, use_epinorminf = false)
expdesignJuMP16() = expdesignJuMP(20, 40, 80, 5, geomean_obj = true) # other big size difficult for this formulation
expdesignJuMP17() = expdesignJuMP(10, 30, 50, 5, geomean_obj = true)
expdesignJuMP18() = expdesignJuMP(5, 15, 25, 5, geomean_obj = true)
expdesignJuMP19() = expdesignJuMP(4, 8, 12, 3, geomean_obj = true)
expdesignJuMP20() = expdesignJuMP(3, 5, 7, 2, geomean_obj = true)
expdesignJuMP21() = expdesignJuMP(25, 75, 125, 5, rootdet_obj = true, use_nat = false)
expdesignJuMP22() = expdesignJuMP(10, 30, 50, 5, rootdet_obj = true, use_nat = false)
expdesignJuMP23() = expdesignJuMP(5, 15, 25, 5, rootdet_obj = true, use_nat = false)
expdesignJuMP24() = expdesignJuMP(4, 8, 12, 3, rootdet_obj = true, use_nat = false)
expdesignJuMP25() = expdesignJuMP(3, 5, 7, 2, rootdet_obj = true, use_nat = false)

function test_expdesignJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
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
    expdesignJuMP8,
    expdesignJuMP9,
    expdesignJuMP10,
    expdesignJuMP11,
    expdesignJuMP12,
    expdesignJuMP13,
    expdesignJuMP14,
    expdesignJuMP15,
    expdesignJuMP16,
    expdesignJuMP17,
    expdesignJuMP18,
    expdesignJuMP19,
    expdesignJuMP20,
    expdesignJuMP21,
    expdesignJuMP22,
    expdesignJuMP23,
    expdesignJuMP24,
    expdesignJuMP25,
    ], options = options)

test_expdesignJuMP(; options...) = test_expdesignJuMP.([
    expdesignJuMP3,
    expdesignJuMP8,
    expdesignJuMP13,
    expdesignJuMP18,
    expdesignJuMP23,
    ], options = options)
