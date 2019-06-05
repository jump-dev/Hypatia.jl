#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

compute a gram matrix of a polynomial, minimizing the logarithm of its determinant
https://perso.uclouvain.be/paul.vandooren/publications/GeninNV99b.pdf

=#

import DynamicPolynomials
const DP = DynamicPolynomials
import JuMP
import PolyJuMP
import MathOptInterface
const MOI = MathOptInterface
import Hypatia
import Random
using LinearAlgebra
using Test

function centralpolymatJuMP(n::Int, halfdeg::Int; use_logdet::Bool = true)
    DP.@polyvar x[1:n]
    monomials = DP.monomials(x, 0:halfdeg)
    L = binomial(n + halfdeg, n)
    coeffs = randn(L, L)
    mat = coeffs * coeffs'
    poly_rand = monomials' * mat * monomials

    model = JuMP.Model()
    JuMP.@variable(model, Q[i in 1:L, 1:L], Symmetric)
    v1 = [Q[i, j] for i in 1:L for j in 1:i] # vectorized Q with correct ordering
    poly_Q = sum(Q[i, j] * monomials[i] * monomials[j] * (i == j ? 1 : 2) for i in 1:L for j in 1:i)
    JuMP.@constraint(model, poly_rand == poly_Q)
    if use_logdet
        JuMP.@variable(model, hypo)
        JuMP.@constraint(model, vcat(hypo, 1, v1) in MOI.LogDetConeTriangle(L))
        JuMP.@objective(model, Max, hypo)
    else
        JuMP.@variable(model, hypo[1:L])
        JuMP.@variable(model, lowertri[i in 1:L, j in 1:i])
        # vectorized loer half of PSD matrix
        v2 = vcat([vcat(zeros(i - 1), [lowertri[j, i] for j in i:L], zeros(i - 1), lowertri[i, i]) for i in 1:L]...)
        JuMP.@constraints(model, begin
            vcat(v1, v2) in MOI.PositiveSemidefiniteConeTriangle(2L)
            [i in 1:L], [hypo[i], 1.0, lowertri[i, i]] in MOI.ExponentialCone()
        end)
        JuMP.@objective(model, Max, sum(hypo))
    end

    return (model = model, poly_rand = poly_rand, poly_Q = poly_Q)
end

centralpolymatJuMP1() = centralpolymatJuMP(2, 3)
centralpolymatJuMP2() = centralpolymatJuMP(3, 2)
centralpolymatJuMP3() = centralpolymatJuMP(2, 3, use_logdet = false)
centralpolymatJuMP4() = centralpolymatJuMP(3, 2, use_logdet = false)

function test_centralpolymatJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    @test d.poly_rand ≈ JuMP.value(d.poly_Q)
    return
end

test_centralpolymatJuMP_all(; options...) = test_centralpolymatJuMP.([
    centralpolymatJuMP1,
    centralpolymatJuMP2,
    centralpolymatJuMP3,
    centralpolymatJuMP4,
    ], options = options)

test_centralpolymatJuMP(; options...) = test_centralpolymatJuMP.([
    centralpolymatJuMP1,
    centralpolymatJuMP2,
    centralpolymatJuMP3,
    centralpolymatJuMP4,
    ], options = options)
