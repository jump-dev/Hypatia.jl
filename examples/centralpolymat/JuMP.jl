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

example_tests(::Type{CentralPolyMatJuMP{Float64}}, ::MinimalInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((2, 2, true), nothing, options),
    ((2, 2, true), ClassicConeOptimizer, options),
    ((2, 2, false), nothing, options),
    ((2, 2, false), ClassicConeOptimizer, options),
    ]
end
example_tests(::Type{CentralPolyMatJuMP{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((2, 3, true), nothing, options),
    ((2, 3, true), ClassicConeOptimizer, options),
    ((2, 3, false), nothing, options),
    ((2, 3, false), ClassicConeOptimizer, options),
    ((3, 2, true), nothing, options),
    ((3, 2, false), nothing, options),
    ((3, 4, true), nothing, options),
    ((3, 4, false), nothing, options),
    ((7, 2, true), nothing, options),
    ((7, 2, true), ClassicConeOptimizer, options),
    ((7, 2, false), nothing, options),
    ((7, 2, false), ClassicConeOptimizer, options),
    ]
end
example_tests(::Type{CentralPolyMatJuMP{Float64}}, ::SlowInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    ((3, 4, true), ClassicConeOptimizer, options),
    ((3, 4, false), ClassicConeOptimizer, options),
    ((3, 5, true), nothing, options),
    ((3, 5, false), nothing, options),
    ((6, 3, true), nothing, options),
    ((6, 3, false), nothing, options),
    ]
end

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

return CentralPolyMatJuMP
