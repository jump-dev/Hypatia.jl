#=
compute a maximum-determinant gram matrix of a polynomial
=#

import DynamicPolynomials

struct CentralPolyMatJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int # number of polyvars
    halfdeg::Int # half degree of random polynomials
    ext::MatSpecExt # formulation specifier
end

function build(inst::CentralPolyMatJuMP{T}) where {T <: Float64}
    (n, halfdeg) = (inst.n, inst.halfdeg)

    DynamicPolynomials.@polyvar x[1:n]
    basis = DynamicPolynomials.monomials(x, 0:halfdeg)
    L = length(basis)
    poly_half = randn(L, L) / L * basis
    poly_rand = poly_half' * poly_half

    model = JuMP.Model()
    JuMP.@variable(model, Q_vec[1:Cones.svec_length(L)])

    # convex objective
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    add_homog_spectral(inst.ext, L, vcat(1.0 * epi, Q_vec), model)

    # coefficients equal
    Q = Symmetric(get_smat_U(L, 1.0 * Q_vec), :U)
    poly_eq = basis' * Q * basis - poly_rand
    JuMP.@constraint(model, DynamicPolynomials.coefficients(poly_eq) .== 0)

    return model
end
