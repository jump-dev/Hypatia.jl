#=
estimate a covariance matrix that satisfies some given prior information and
minimizes a given convex spectral function

P ∈ 𝕊ᵈ is the covariance variable
minimize    f(P)                    (note: enforces P ⪰ 0)
subject to  tr(P) = 1               (normalize)
            B vec(P) = b            (prior info as equalities)
            C vec(P) ≤ c            (prior info as inequalities)
where f is a convex spectral function
=#

struct CovarianceEstJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
    ext::MatSpecExt # formulation specifier
end

function build(inst::CovarianceEstJuMP{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 1
    @assert is_domain_pos(inst.ext)

    P0 = randn(T, d, d)
    P0 = Hermitian(P0 * P0' + 0.5 * I, :U)
    P0 *= d / tr(P0)
    vec_dim = Cones.svec_length(d)
    P0_vec = zeros(T, vec_dim)
    Cones.smat_to_svec!(P0_vec, P0, sqrt(T(2)))

    model = JuMP.Model()
    JuMP.@variable(model, P[1:d, 1:d], Symmetric)
    JuMP.@constraint(model, tr(P) == d)
    P_vec = zeros(JuMP.AffExpr, vec_dim)
    Cones.smat_to_svec!(P_vec, one(T) * P, sqrt(T(2)))

    # convex objective
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    add_homog_spectral(inst.ext, d, vcat(1.0 * epi, P_vec), model)

    # linear prior constraints
    lin_dim = round(Int, sqrt(d - 1))
    B = randn(T, lin_dim, vec_dim)
    b = B * P0_vec
    JuMP.@constraint(model, B * P_vec .== b)
    C = randn(T, lin_dim, vec_dim)
    c = C * P0_vec
    JuMP.@constraint(model, C * P_vec .<= c)

    # save for use in tests
    model.ext[:P_var] = P

    return model
end

function test_extra(inst::CovarianceEstJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective
    tol = eps(T)^0.2
    P_opt = JuMP.value.(model.ext[:P_var])
    λ = eigvals(Symmetric(P_opt, :U))
    @test minimum(λ) >= -tol
    obj_result = get_val(pos_only(λ), inst.ext)
    @test JuMP.objective_value(model) ≈ obj_result atol=tol rtol=tol
    return
end
