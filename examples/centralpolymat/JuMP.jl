#=
minimize a spectral function of a gram matrix of a polynomial
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
    Q0 = randn(L, L)
    if is_domain_pos(inst.ext)
        # make the polynomial nonnegative
        Q0 = Q0' * Q0
    end
    Q0 .*= inv(L)
    poly = basis' * Symmetric(Q0, :U) * basis

    model = JuMP.Model()
    JuMP.@variable(model, Q_vec[1:Cones.svec_length(L)])

    # convex objective
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    add_homog_spectral(inst.ext, L, vcat(1.0 * epi, Q_vec), model)

    # coefficients equal
    Q = Symmetric(get_smat_U(L, 1.0 * Q_vec), :U)
    poly_eq = basis' * Q * basis - poly
    JuMP.@constraint(model, DynamicPolynomials.coefficients(poly_eq) .== 0)

    # save for use in tests
    model.ext[:Q_var] = Q

    return model
end

function test_extra(inst::CentralPolyMatJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective
    tol = eps(T)^0.2
    Q_opt = JuMP.value.(model.ext[:Q_var])
    λ = eigvals(Symmetric(Q_opt, :U))
    if is_domain_pos(inst.ext)
        @test minimum(λ) >= -tol
        λ = pos_only(λ)
    end
    obj_result = get_val(λ, inst.ext)
    @test JuMP.objective_value(model) ≈ obj_result atol=tol rtol=tol
    return
end
