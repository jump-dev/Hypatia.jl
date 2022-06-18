#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
given a random X variable taking values in the finite set {α₁,...,αₙ}, compute
the distribution minimizing a given convex spectral function over all distributions
satisfying some prior information (expressed using equality constraints)

adapted from Boyd and Vandenberghe, "Convex Optimization", section 7.2

p ∈ ℝᵈ is the probability variable
minimize    f(p)            (note: enforces p ≥ 0)
subject to  Σᵢ pᵢ = d       (probability distribution, scaled by d)
            A p = b         (prior info)
where f is a convex spectral function
=#

struct NonparametricDistrJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    d::Int
    ext::VecSpecExt # formulation specifier
end

function build(inst::NonparametricDistrJuMP{T}) where {T <: Float64}
    d = inst.d
    @assert d >= 2
    @assert is_domain_pos(inst.ext) # domain must be positive
    p0 = rand(T, d)
    p0 .*= d / sum(p0)

    model = JuMP.Model()
    JuMP.@variable(model, p[1:d])
    JuMP.@constraint(model, sum(p) == d)

    # linear prior constraints
    A = randn(T, round(Int, d / 2), d)
    b = A * p0
    JuMP.@constraint(model, A * p .== b)

    # convex objective
    JuMP.@variable(model, epi)
    JuMP.@objective(model, Min, epi)
    add_homog_spectral(inst.ext, d, vcat(1.0 * epi, p), model)

    # save for use in tests
    model.ext[:p_var] = p

    return model
end

function test_extra(inst::NonparametricDistrJuMP{T}, model::JuMP.Model) where {T}
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # check objective and constraints
    tol = eps(T)^0.2
    p_opt = JuMP.value.(model.ext[:p_var])
    @test sum(p_opt) ≈ inst.d atol = tol rtol = tol
    @test minimum(p_opt) >= -tol
    p_opt = pos_only(p_opt)
    obj_result = get_val(p_opt, inst.ext)
    @test JuMP.objective_value(model) ≈ obj_result atol = tol rtol = tol
    return
end
