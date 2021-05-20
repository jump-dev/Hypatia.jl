#=
estimate a covariance matrix (PSD) minimizing a given convex spectral function
and satisfying some prior information

p ∈ 𝕊ᵈ is the covariance variable
minimize    f(p)                    (note: enforces p ⪰ 0)
subject to  tr(p) = 1               (normalize)
            B p = b                 (prior info as equalities)
            C p ≤ c                 (prior info as inequalities)
where f is a convex spectral function
=#

struct CovarianceEstJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
    ssf::Cones.SepSpectralFun
end

function build(inst::CovarianceEstJuMP{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 1
    p0 = randn(T, d, d)
    p0 = p0 * p0' + I / 2
    p0 ./= tr(p0)
    vec_dim = Cones.svec_length(d)
    p0_vec = zeros(T, vec_dim)
    Cones.smat_to_svec!(p0_vec, p0, sqrt(T(2)))

    model = JuMP.Model()
    JuMP.@variable(model, p[1:d, 1:d], Symmetric)
    JuMP.@constraint(model, tr(p) == 1)
    p_vec = zeros(JuMP.AffExpr, vec_dim)
    Cones.smat_to_svec!(p_vec, one(T) * p, sqrt(T(2)))

    # convex objective
    f_cone = Hypatia.EpiPerSepSpectralCone{T}(inst.ssf, Cones.MatrixCSqr{T, T}, d)
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    JuMP.@constraint(model, vcat(epi, 1, p_vec) in f_cone)

    # linear prior constraints
    lin_dim = round(Int, sqrt(d - 1))
    B = randn(T, lin_dim, vec_dim)
    b = B * p0_vec
    JuMP.@constraint(model, B * p_vec .== b)
    C = randn(T, lin_dim, vec_dim)
    c = C * p0_vec
    JuMP.@constraint(model, C * p_vec .<= c)

    return model
end
