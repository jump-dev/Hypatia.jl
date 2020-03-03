#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

compute a gram matrix of a polynomial, minimizing its log-determinant or root-determinant (equivalent optimal solutions with different optimal objective values)
=#

import DynamicPolynomials
const DP = DynamicPolynomials
import JuMP
const MOI = JuMP.MOI
import Hypatia
import Random
using LinearAlgebra
using Test

function centralpolymat_JuMP(
    ::Type{T},
    n::Int,
    halfdeg::Int,
    logdet_obj::Bool, # use logdet, else rootdet
    ) where {T <: Float64} # TODO support generic reals
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

    JuMP.@variable(model, hypo)
    JuMP.@objective(model, Max, hypo)
    if logdet_obj
        JuMP.@constraint(model, vcat(hypo, 1.0, v1) in MOI.LogDetConeTriangle(L))
    else
        JuMP.@constraint(model, vcat(hypo, v1) in MOI.RootDetConeTriangle(L))
    end

    return (model = model,)
end

function test_centralpolymat_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = centralpolymat_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

centralpolymat_JuMP_fast = [
    (2, 3, true),
    (2, 3, false),
    (3, 2, true),
    (3, 2, false),
    (3, 4, true),
    (3, 4, false),
    (7, 2, true),
    (7, 2, false),
    ]
centralpolymat_JuMP_slow = [
    (3, 5, true),
    (3, 5, false),
    (6, 3, true),
    (6, 3, false),
    ]
