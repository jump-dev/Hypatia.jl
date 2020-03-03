#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in examples/polymin/native.jl
=#

import Random
using LinearAlgebra
using Test
import JuMP
import Hypatia
const MU = Hypatia.ModelUtilities

include(joinpath(@__DIR__, "data.jl"))

function polymin_JuMP(
    T::Type{Float64}, # TODO support generic reals
    interp_vals::Vector{T},
    Ps::Vector{Matrix{T}},
    true_min::Real,
    use_primal::Bool, # solve primal, else solve dual
    use_wsos::Bool, # use wsosinterpnonnegative cone, else PSD formulation
    )
    U = length(interp_vals)

    model = JuMP.Model()
    if use_primal
        JuMP.@variable(model, a)
        JuMP.@objective(model, Max, a)
    else
        JuMP.@variable(model, μ[1:U])
        JuMP.@objective(model, Min, dot(μ, interp_vals))
        JuMP.@constraint(model, sum(μ) == 1.0) # TODO can remove this constraint and a variable
    end

    if use_wsos
        cone = Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps, !use_primal)
        aff_expr = (use_primal ? interp_vals .- a : μ)
        JuMP.@constraint(model, aff_expr in cone)
    else
        if use_primal
            psd_vars = []
            for (k, P) in enumerate(Ps)
                Lk = size(P, 2)
                psd_k = JuMP.@variable(model, [1:Lk, 1:Lk], Symmetric)
                push!(psd_vars, psd_k)
                JuMP.@SDconstraint(model, psd_k >= 0)
            end
            JuMP.@constraint(model, sum(diag(P * psd_k * P') for (P, psd_k) in zip(Ps, psd_vars)) .== interp_vals .- a)
        else
            for P in Ps
                L = size(P, 2)
                JuMP.@constraint(model, [sum(P[u, i] * P[u, j] * μ[u] for u in 1:U) for i in 1:L for j in 1:i] in JuMP.MOI.PositiveSemidefiniteConeTriangle(L))
            end
        end
    end

    return (model = model, true_min = true_min)
end

polymin_JuMP(
    T::Type{<:Real},
    poly_name::Symbol,
    halfdeg::Int,
    args...
    ) = polymin_JuMP(T, get_interp_data(T, poly_name, halfdeg)..., args...)

polymin_JuMP(
    T::Type{<:Real},
    n::Int,
    halfdeg::Int,
    args...
    ) = polymin_JuMP(T, random_interp_data(T, n, halfdeg)..., args...)

function test_polymin_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = polymin_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    if !isnan(d.true_min)
        @test JuMP.objective_value(d.model) ≈ d.true_min atol = 1e-4 rtol = 1e-4
    end
    return d.model.moi_backend.optimizer.model.optimizer.result
end

polymin_JuMP_fast = [
    (:butcher, 2, true, true),
    (:caprasse, 4, true, true),
    (:goldsteinprice, 7, true, true),
    (:goldsteinprice_ball, 7, true, true),
    (:goldsteinprice_ellipsoid, 7, true, true),
    (:heart, 2, true, true),
    (:lotkavolterra, 3, true, true),
    (:magnetism7, 2, true, true),
    (:magnetism7_ball, 2, true, true),
    (:motzkin, 3, true, true),
    (:motzkin_ball, 3, true, true),
    (:motzkin_ellipsoid, 3, true, true),
    (:reactiondiffusion, 4, true, true),
    (:robinson, 8, true, true),
    (:robinson_ball, 8, true, true),
    (:rosenbrock, 5, true, true),
    (:rosenbrock_ball, 5, true, true),
    (:schwefel, 2, true, true),
    (:schwefel_ball, 2, true, true),
    (:lotkavolterra, 3, false, true),
    (:motzkin, 3, false, true),
    (:motzkin_ball, 3, false, true),
    (:schwefel, 2, false, true),
    (:lotkavolterra, 3, false, false),
    (:motzkin, 3, false, false),
    (:motzkin_ball, 3, false, false),
    (:schwefel, 2, false, false),
    (1, 8, true, true),
    (2, 5, true, true),
    (3, 3, true, true),
    (5, 2, true, true),
    (3, 3, false, true),
    (3, 3, false, false),
    ]
polymin_JuMP_slow = [
    # TODO
    ]
