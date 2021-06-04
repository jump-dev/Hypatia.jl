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
    num_spec::Int # number of spectral cones
    use_EFs::Bool # use standard cone extended formulations for spectral cones
end

function build(inst::NonparametricDistrJuMP{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 2
    p0 = rand(T, d)
    p0 .*= d / sum(p0)

    # pick random spectral cones (or EFs)
    if inst.use_EFs
        exts = [VecNegGeomEFExp(), VecNegGeomEFPow(), VecInvEF(), VecNegLogEF(),
            VecNegEntropyEF(), VecPower12EF(1.5)]
    else
        exts = [VecNegGeom(), VecInv(), VecNegLog(), VecNegEntropy(),
            VecPower12(1.5)]
    end
    @assert 1 <= inst.num_spec <= length(exts)
    exts = Random.shuffle!(exts)[1:inst.num_spec]

    model = JuMP.Model()
    JuMP.@variable(model, p[1:d])
    JuMP.@constraint(model, sum(p) == d)

    # linear prior constraints
    B = randn(T, round(Int, sqrt(d - 1)), d)
    b = B * p0
    JuMP.@constraint(model, B * p .== b)
    C = randn(T, round(Int, log(d - 1)), d)
    c = C * p0
    JuMP.@constraint(model, C * p .<= c)

    # convex objective
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    add_homog_spectral(exts[1], d, vcat(1.0 * epi, p), model)

    # convex constraints
    val_p0s = T[]
    for ext in exts[2:end]
        val_p0 = get_val(p0, ext)
        push!(val_p0s, val_p0)
        add_homog_spectral(ext, d, vcat(val_p0, p), model)
    end

    # save for use in tests
    model.ext[:exts] = exts
    model.ext[:val_p0s] = val_p0s
    model.ext[:p_var] = p

    return model
end

function test_extra(inst::NonparametricDistrJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective and constraints
    tol = eps(T)^0.20
    exts = model.ext[:exts]
    val_p0s = model.ext[:val_p0s]
    p_opt = JuMP.value.(model.ext[:p_var])
    d = length(p_opt)
    @test sum(p_opt) ≈ d atol=tol rtol=tol
    # objective
    obj_result = get_val(p_opt, exts[1])
    @test JuMP.objective_value(model) ≈ obj_result atol=tol rtol=tol
    # convex constraints
    for (i, ext) in enumerate(exts[2:end])
        @test val_p0s[i] >= get_val(p_opt, ext) - tol
    end
    return
end
