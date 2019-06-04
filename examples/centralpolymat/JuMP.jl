#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

compute a gram matrix of a polynomial, minimizing the logarithm of its determinant
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

function centralpolymatJuMP(n::Int, d::Int)
    DP.@polyvar x[1:n]
    halfdeg_monos = DP.monomials(x, 0:d)
    L = binomial(n + d, n)
    U = binomial(n + 2d, n)
    coeffs = randn(L, L)
    mat = coeffs * coeffs'
    poly = halfdeg_monos' * mat * halfdeg_monos

    model = JuMP.Model()
    JuMP.@variables(model, begin
        Q[1:L, 1:L], Symmetric
        z
    end)
    JuMP.@constraint(model, poly == halfdeg_monos' * Q * halfdeg_monos)
    JuMP.@constraint(model, [z; 1; [Q[i, j] for i in 1:L for j in 1:i]] in MOI.LogDetConeTriangle(L))
    JuMP.@objective(model, Max, z)

    return (model = model, mat = mat, monomials = halfdeg_monos, Q = Q)
end

centralpolymatJuMP1() = centralpolymatJuMP(2, 3)
centralpolymatJuMP2() = centralpolymatJuMP(3, 2)


function test_centralpolymatJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    @test d.monomials' * JuMP.value.(d.Q) * d.monomials ≈  d.monomials' * d.mat * d.monomials
    return
end

test_centralpolymatJuMP_all(; options...) = test_centralpolymatJuMP.([
    centralpolymatJuMP1,
    centralpolymatJuMP2,
    ], options = options)

test_centralpolymatJuMP(; options...) = test_centralpolymatJuMP.([
    centralpolymatJuMP1,
    centralpolymatJuMP2,
    ], options = options)
