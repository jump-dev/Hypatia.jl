#=
given a random X variable taking values in the finite set {α₁,...,αₙ}, compute
the distribution minimizing a given convex spectral function over all distributions
satisfying some prior information (expressed using convex constraints)

adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.2

p ∈ ℝᵈ is the probability variable, scaled by d (to keep each pᵢ close to 1)
minimize    f(p)            (note: enforces p ≥ 0)
subject to  Σᵢ pᵢ = d       (probability distribution, scaled by d)
            gⱼ(p) ≤ kⱼ ∀j   (prior info as convex constraints)
            B p = b         (prior info as equalities)
            C p ≤ c         (prior info as inequalities)
where f and gⱼ are different convex spectral functions
=#

struct NonparametricDistrJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
end

function build(inst::NonparametricDistrJuMP{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 2
    p0 = rand(T, d)
    p0 .*= d / sum(p0)

    fg_funs = Random.shuffle!([
        Cones.InvSSF(),
        Cones.NegLogSSF(),
        Cones.NegEntropySSF(),
        Cones.Power12SSF(1.5),
        ])

    model = JuMP.Model()
    JuMP.@variable(model, p[1:d])
    JuMP.@constraint(model, sum(p) == d)

    # convex objective
    f_cone = Hypatia.EpiPerSepSpectralCone{T}(fg_funs[1], Cones.VectorCSqr{T}, d)
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    JuMP.@constraint(model, vcat(epi, 1, p) in f_cone)

    # convex prior constraints
    for g in fg_funs[2:end]
        g_cone = Hypatia.EpiPerSepSpectralCone{T}(g, Cones.VectorCSqr{T}, d)
        k = Cones.h_val(p0, g)
        JuMP.@constraint(model, vcat(k, 1, p) in g_cone)
    end

    # linear prior constraints
    B = randn(T, round(Int, sqrt(d - 1)), d)
    b = B * p0
    JuMP.@constraint(model, B * p .== b)
    C = randn(T, round(Int, log(d - 1)), d)
    c = C * p0
    JuMP.@constraint(model, C * p .<= c)

    return model
end
