#=
compute a maximum-determinant gram matrix of a polynomial
=#

import DynamicPolynomials

struct CentralPolyMatJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int # number of polyvars
    halfdeg::Int # half degree of random polynomials
    logdet_obj::Bool # use logdet, else rootdet
end

function build(inst::CentralPolyMatJuMP{T}) where {T <: Float64}
    (n, halfdeg) = (inst.n, inst.halfdeg)

    DynamicPolynomials.@polyvar x[1:n]
    basis = DynamicPolynomials.monomials(x, 0:halfdeg)
    L = length(basis)
    poly_half = randn(L, L) / L * basis
    poly_rand = poly_half' * poly_half

    model = JuMP.Model()
    JuMP.@variable(model, hypo)
    JuMP.@objective(model, Max, hypo)

    # objective hypograph
    JuMP.@variable(model, Q[1:L, 1:L], Symmetric)
    v1 = [Q[i, j] for i in 1:L for j in 1:i] # vectorized Q
    if inst.logdet_obj
        JuMP.@constraint(model, vcat(hypo, 1, v1) in MOI.LogDetConeTriangle(L))
    else
        JuMP.@constraint(model, vcat(hypo, v1) in MOI.RootDetConeTriangle(L))
    end

    # coefficients equal
    poly_Q = basis' * Q * basis
    JuMP.@constraint(model,
        DynamicPolynomials.coefficients(poly_Q - poly_rand) .== 0)

    return model
end
