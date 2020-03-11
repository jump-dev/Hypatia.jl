#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

compute a gram matrix of a polynomial, minimizing its log-determinant or root-determinant (equivalent optimal solutions with different optimal objective values)
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
import DynamicPolynomials
const DP = DynamicPolynomials

struct CentralPolyMatJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int # number of polyvars
    halfdeg::Int # half degree of random polynomials
    logdet_obj::Bool # use logdet, else rootdet
end

options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
example_tests(::Type{CentralPolyMatJuMP{Float64}}, ::MinimalInstances) = [
    ((2, 2, true), false, options),
    ((2, 2, false), false, options),
    # ((2, 2, true), true, options),
    # ((2, 2, false), true, options),
    ]
example_tests(::Type{CentralPolyMatJuMP{Float64}}, ::FastInstances) = [
    ((2, 3, true), false, options),
    ((2, 3, false), false, options),
    ((3, 2, true), false, options),
    ((3, 2, false), false, options),
    ((3, 4, true), false, options),
    ((3, 4, false), false, options),
    ((7, 2, true), false, options),
    ((7, 2, false), false, options),
    ]
example_tests(::Type{CentralPolyMatJuMP{Float64}}, ::SlowInstances) = [
    ((3, 5, true), false, options),
    ((3, 5, false), false, options),
    ((6, 3, true), false, options),
    ((6, 3, false), false, options),
    ]

function build(inst::CentralPolyMatJuMP{T}) where {T <: Float64} # TODO generic reals
    (n, halfdeg) = (inst.n, inst.halfdeg)

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
    if inst.logdet_obj
        JuMP.@constraint(model, vcat(hypo, 1.0, v1) in MOI.LogDetConeTriangle(L))
    else
        JuMP.@constraint(model, vcat(hypo, v1) in MOI.RootDetConeTriangle(L))
    end

    return model
end

function test_extra(inst::CentralPolyMatJuMP, model, options)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

# @testset "CentralPolyMatJuMP" for inst in example_tests(CentralPolyMatJuMP{Float64}, MinimalInstances()) test(inst...) end

return CentralPolyMatJuMP
