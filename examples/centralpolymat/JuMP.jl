#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

compute a gram matrix of a polynomial, minimizing its log-determinant or root-determinant (equivalent optimal solutions with different optimal objective values)
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
import DynamicPolynomials
const DP = DynamicPolynomials

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

    return (model, ())
end

function test_centralpolymat_JuMP(model, test_helpers, test_options)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
centralpolymat_JuMP_fast = [
    ((Float64, 2, 3, true), false, (), options),
    ((Float64, 2, 3, false), false, (), options),
    ((Float64, 3, 2, true), false, (), options),
    ((Float64, 3, 2, false), false, (), options),
    ((Float64, 3, 4, true), false, (), options),
    ((Float64, 3, 4, false), false, (), options),
    ((Float64, 7, 2, true), false, (), options),
    ((Float64, 7, 2, false), false, (), options),
    ]
centralpolymat_JuMP_slow = [
    ((Float64, 3, 5, true), false, (), options),
    ((Float64, 3, 5, false), false, (), options),
    ((Float64, 6, 3, true), false, (), options),
    ((Float64, 6, 3, false), false, (), options),
    ]

@testset "centralpolymat_JuMP" begin test_JuMP_instance.(centralpolymat_JuMP, test_centralpolymat_JuMP, centralpolymat_JuMP_fast) end
;
