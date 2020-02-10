#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

compute a gram matrix of a polynomial, minimizing its log-determinant or root-determinant (equivalent optimal solutions with different optimal objective values)
=#

import DynamicPolynomials
const DP = DynamicPolynomials
import JuMP
import MathOptInterface
const MOI = MathOptInterface
import Hypatia
import Random
using LinearAlgebra
using Test

function centralpolymatJuMP(
    n::Int,
    halfdeg::Int;
    logdet_obj::Bool = true, # use logdet, else rootdet
    use_natural::Bool = true, # use natural, else extended
    )
    DP.@polyvar x[1:n]
    monomials = DP.monomials(x, 0:halfdeg)
    L = binomial(n + halfdeg, n)
    coeffs = randn(L, L)
    mat = coeffs * coeffs'
    poly_rand = monomials' * mat * monomials

    model = JuMP.Model()
    JuMP.@variable(model, Q[i in 1:L, 1:L], Symmetric)
    v1 = [Q[i, j] for i in 1:L for j in 1:i] # vectorized Q
    poly_Q = sum(Q[i, j] * monomials[i] * monomials[j] * (i == j ? 1 : 2) for i in 1:L for j in 1:i)
    JuMP.@constraint(model, DP.coefficients(poly_Q - poly_rand) .== 0)

    if use_natural
        JuMP.@variable(model, hypo)
        JuMP.@objective(model, Max, hypo)
        if logdet_obj
            JuMP.@constraint(model, vcat(hypo, 1.0, v1) in MOI.LogDetConeTriangle(L))
        else
            JuMP.@constraint(model, vcat(hypo, v1) in MOI.RootDetConeTriangle(L))
        end
    else
        JuMP.@variable(model, lowertri[i in 1:L, j in 1:i])
        v2 = vcat([vcat(zeros(i - 1), [lowertri[j, i] for j in i:L], zeros(i - 1), lowertri[i, i]) for i in 1:L]...)
        JuMP.@constraint(model, vcat(v1, v2) in MOI.PositiveSemidefiniteConeTriangle(2L))
        if logdet_obj
            JuMP.@variable(model, hypo[1:L])
            JuMP.@objective(model, Max, sum(hypo))
            JuMP.@constraint(model, [i in 1:L], [hypo[i], 1.0, lowertri[i, i]] in MOI.ExponentialCone())
        else
            JuMP.@variable(model, hypo)
            JuMP.@objective(model, Max, hypo)
            JuMP.@constraint(model, vcat(hypo, [lowertri[i, i] for i in 1:L]) in MOI.GeometricMeanCone(L + 1))
        end
    end

    return (model = model,)
end

# TODO add larger sizes
centralpolymatJuMP1() = centralpolymatJuMP(2, 3, logdet_obj = true, use_natural = true)
centralpolymatJuMP2() = centralpolymatJuMP(3, 2, logdet_obj = true, use_natural = true)
centralpolymatJuMP3() = centralpolymatJuMP(2, 3, logdet_obj = false, use_natural = true)
centralpolymatJuMP4() = centralpolymatJuMP(3, 2, logdet_obj = false, use_natural = true)
centralpolymatJuMP5() = centralpolymatJuMP(2, 3, logdet_obj = true, use_natural = false)
centralpolymatJuMP6() = centralpolymatJuMP(3, 2, logdet_obj = true, use_natural = false)
centralpolymatJuMP7() = centralpolymatJuMP(2, 3, logdet_obj = false, use_natural = false)
centralpolymatJuMP8() = centralpolymatJuMP(3, 2, logdet_obj = false, use_natural = false)

function test_centralpolymatJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_centralpolymatJuMP_all(; options...) = test_centralpolymatJuMP.([
    centralpolymatJuMP1,
    centralpolymatJuMP2,
    centralpolymatJuMP3,
    centralpolymatJuMP4,
    centralpolymatJuMP5,
    centralpolymatJuMP6,
    centralpolymatJuMP7,
    centralpolymatJuMP8,
    ], options = options)

test_centralpolymatJuMP(; options...) = test_centralpolymatJuMP.([
    centralpolymatJuMP1,
    centralpolymatJuMP2,
    centralpolymatJuMP3,
    centralpolymatJuMP4,
    centralpolymatJuMP5,
    centralpolymatJuMP6,
    centralpolymatJuMP7,
    centralpolymatJuMP8,
    ], options = options)
