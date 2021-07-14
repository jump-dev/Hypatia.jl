#=
given a random X variable taking values in the finite set {α₁,...,αₙ}, compute
the distribution minimizing a given convex spectral function over all distributions
satisfying some prior information (expressed using equality constraints)

adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.2

p ∈ ℝᵈ is the probability variable
minimize    f(p)            (note: enforces p ≥ 0)
subject to  Σᵢ pᵢ = d       (probability distribution, scaled by d)
            gⱼ(D p) ≤ kⱼ ∀j (prior info as convex constraints)
            A p = b         (prior info as equalities)
where f and gⱼ are different convex spectral functions
=#

struct NonparametricDistrJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
    exts::Vector{VecSpecExt} # formulation specifier
end

function build(inst::NonparametricDistrJuMP{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 2
    exts = inst.exts
    @assert length(exts) >= 1 # first is for objective
    @assert all(is_domain_pos, exts) # domain must be positive
    p0 = rand(T, d)
    p0 .*= d / sum(p0)

    model = JuMP.Model()
    JuMP.@variable(model, p[1:d])
    JuMP.@constraint(model, sum(p) == d)

    # linear prior constraints
    A = randn(T, round(Int, d / 3), d)
    b = A * p0
    JuMP.@constraint(model, A * p .== b)

    # convex objective
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    add_homog_spectral(exts[1], d, vcat(1.0 * epi, p), model)

    # convex constraints
    con_aff = Vector{Tuple{T, Matrix{T}}}()
    for ext in exts[2:end]
        D = rand(T, d, d)
        val_p0 = get_val(D * p0, ext)
        push!(con_aff, (val_p0, D))
        add_homog_spectral(ext, d, vcat(val_p0, D * p), model)
    end

    # save for use in tests
    model.ext[:con_aff] = con_aff
    model.ext[:p_var] = p

    return model
end

function test_extra(inst::NonparametricDistrJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective and constraints
    tol = eps(T)^0.2
    exts = inst.exts
    con_aff = model.ext[:con_aff]
    p_opt = JuMP.value.(model.ext[:p_var])
    @test sum(p_opt) ≈ inst.d atol=tol rtol=tol
    @test minimum(p_opt) >= -tol
    p_opt = pos_only(p_opt)
    obj_result = get_val(p_opt, exts[1])
    @test JuMP.objective_value(model) ≈ obj_result atol=tol rtol=tol
    for (i, ext) in enumerate(exts[2:end])
        (val_p0, D) = con_aff[i]
        @test val_p0 >= get_val(D * p_opt, ext) - tol
    end
    return
end
