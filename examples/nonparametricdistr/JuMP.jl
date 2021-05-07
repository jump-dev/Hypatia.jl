#=
given a random X variable taking values in the finite set {α₁,...,αₙ}, compute
the distribution minimizing a given convex spectral function over all distributions
satisfying some prior information (expressed using convex constraints)

adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.2

p ∈ ℝᵈ is the probability variable
minimize    f(p)                    (note: enforces p ≥ 0)
subject to  Σᵢ pᵢ = 1               (probability distribution)
            gⱼ(Aⱼ p + aⱼ) ≤ kⱼ ∀j   (prior info as convex constraints)
            B p = b                 (prior info as equalities)
            C p ≤ c                 (prior info as inequalities)
where f and gⱼ are spectral functions
=#

struct NonparametricDistrJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
end

function build(inst::NonparametricDistrJuMP{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 1
    p0 = rand(T, d)
    p0 ./= sum(p0)

    sep_spectral_funs = [
        Cones.InvSSF(),
        Cones.NegLogSSF(),
        Cones.NegEntropySSF(),
        Cones.Power12SSF(1.5),
        ]

    model = JuMP.Model()
    JuMP.@variable(model, p[1:d])
    JuMP.@constraint(model, sum(p) == 1)

    # convex objective
    f = rand(sep_spectral_funs)
    f_cone = Hypatia.EpiPerSepSpectralCone{T}(f, Cones.VectorCSqr{T}, d)
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    JuMP.@constraint(model, vcat(epi, 1, p) in f_cone)

    # convex prior constraints
    for g in sep_spectral_funs
        A = randn(T, d, d)
        a0 = 1 .+ T(0.1) * randn(T, d)
        a = a0 - A * p0
        k = Cones.h_val(a0, g)
        g_cone = Hypatia.EpiPerSepSpectralCone{T}(g, Cones.VectorCSqr{T}, d)
        JuMP.@constraint(model, vcat(k, 1, A * p + a) in g_cone)
    end

    # linear prior constraints
    lin_dim = round(Int, sqrt(d - 1))
    B = randn(T, lin_dim, d)
    b = B * p0
    JuMP.@constraint(model, B * p .== b)
    C = randn(T, lin_dim, d)
    c = C * p0
    JuMP.@constraint(model, C * p .<= c)

    return model
end
