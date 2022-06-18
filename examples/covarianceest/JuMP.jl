#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

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
    complex::Bool
    ext::MatSpecExt # formulation specifier
end

function build(inst::CovarianceEstJuMP{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 1
    @assert is_domain_pos(inst.ext)
    rt2 = sqrt(T(2))
    R = (inst.complex ? Complex{T} : T)

    P0 = randn(R, d, d)
    P0 = Hermitian(P0 * P0' + 0.5 * I, :U)
    P0 *= d / tr(P0)
    vec_dim = Cones.svec_length(R, d)
    P0_vec = zeros(T, vec_dim)
    Cones.smat_to_svec!(P0_vec, P0, rt2)
    B = randn(T, div(d, 3), vec_dim)
    b = B * P0_vec
    C = randn(T, round(Int, log(d)), vec_dim)
    c = C * P0_vec

    model = JuMP.Model()
    JuMP.@variable(model, P_vec[1:vec_dim])

    # convex objective
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    add_homog_spectral(inst.ext, d, vcat(1.0 * epi, P_vec), model)

    # trace constraint
    trP = P_vec[1]
    incr = (inst.complex ? 2 : 1)
    idx = 1
    for i in 1:(d - 1)
        idx += incr * i + 1
        trP += P_vec[idx]
    end
    JuMP.@constraint(model, trP == d)

    # linear prior constraints
    JuMP.@constraint(model, B * P_vec .== b)
    JuMP.@constraint(model, C * P_vec .<= c)

    # save for use in tests
    model.ext[:P_var] = P_vec

    return model
end

function test_extra(inst::CovarianceEstJuMP{T}, model::JuMP.Model) where {T}
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective
    tol = eps(T)^0.2
    P_vec_opt = JuMP.value.(model.ext[:P_var])
    R = (inst.complex ? Complex{T} : T)
    P_opt = Cones.svec_to_smat!(zeros(R, inst.d, inst.d), P_vec_opt, sqrt(T(2)))
    λ = eigvals(Hermitian(P_opt, :U))
    @test minimum(λ) >= -tol
    obj_result = get_val(pos_only(λ), inst.ext)
    @test JuMP.objective_value(model) ≈ obj_result atol = tol rtol = tol
    return
end
