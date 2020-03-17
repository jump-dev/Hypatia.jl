#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

see description in examples/polymin/native.jl
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
include(joinpath(@__DIR__, "data.jl"))

struct PolyMinJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    interp_vals::Vector{T}
    Ps::Vector{Matrix{T}}
    true_min::Real
    use_primal::Bool # solve primal, else solve dual
    use_wsos::Bool # use wsosinterpnonnegative cone, else PSD formulation
end
function PolyMinJuMP{Float64}(
    poly_name::Symbol,
    halfdeg::Int,
    args...)
    return PolyMinJuMP{Float64}(get_interp_data(Float64, poly_name, halfdeg)..., args...)
end
function PolyMinJuMP{Float64}(
    n::Int,
    halfdeg::Int,
    args...)
    return PolyMinJuMP{Float64}(random_interp_data(Float64, n, halfdeg)..., args...)
end

example_tests(::Type{PolyMinJuMP{Float64}}, ::MinimalInstances) = [
    ((1, 2, true, true),),
    ((1, 2, false, true),),
    ((1, 2, false, false),),
    ((:motzkin, 3, true, true),),
    ]
example_tests(::Type{PolyMinJuMP{Float64}}, ::FastInstances) = [
    ((1, 3, true, true),),
    ((1, 30, true, true),),
    ((1, 30, false, true),),
    ((1, 30, false, false),),
    ((2, 8, true, true),),
    ((3, 6, true, true),),
    ((5, 3, true, true),),
    ((10, 1, true, true),),
    ((10, 1, false, true),),
    ((10, 1, false, false),),
    ((4, 4, true, true),),
    ((4, 4, false, true),),
    ((4, 4, false, false),),
    ((:butcher, 2, true, true),),
    ((:caprasse, 4, true, true),),
    ((:goldsteinprice, 7, true, true),),
    ((:goldsteinprice_ball, 6, true, true),),
    ((:goldsteinprice_ellipsoid, 7, true, true),),
    ((:heart, 2, true, true),),
    ((:lotkavolterra, 3, true, true),),
    ((:magnetism7, 2, true, true),),
    ((:magnetism7_ball, 2, true, true),),
    ((:motzkin, 3, true, true),),
    ((:motzkin_ball, 3, true, true),),
    ((:motzkin_ellipsoid, 3, true, true),),
    ((:reactiondiffusion, 4, true, true),),
    ((:robinson, 8, true, true),),
    ((:robinson_ball, 8, true, true),),
    ((:rosenbrock, 5, true, true),),
    ((:rosenbrock_ball, 5, true, true),),
    ((:schwefel, 2, true, true),),
    ((:schwefel_ball, 2, true, true),),
    ((:lotkavolterra, 3, false, true),),
    ((:motzkin, 3, false, true),),
    ((:motzkin_ball, 3, false, true),),
    ((:schwefel, 2, false, true),),
    ((:lotkavolterra, 3, false, false),),
    ((:motzkin, 3, false, false),),
    ((:motzkin_ball, 3, false, false),),
    ]
example_tests(::Type{PolyMinJuMP{Float64}}, ::SlowInstances) = [
    ((4, 5, true, true),),
    ((4, 5, false, true),),
    ((4, 5, false, false),),
    ((2, 30, true, true),),
    ((2, 30, false, true),),
    ((2, 30, false, false),),
    ]

function build(inst::PolyMinJuMP{T}) where {T <: Float64} # TODO generic reals
    (interp_vals, Ps, use_primal) = (inst.interp_vals, inst.Ps, inst.use_primal)
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

    if inst.use_wsos
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
            coeffs_lhs = JuMP.@expression(model, [u in 1:U], sum(sum(P[u, k] * P[u, l] * psd_k[k, l] * (k == l ? 1 : 2) for k in 1:size(Pr, 2) for l in 1:k) for (P, psd_k) in zip(Ps, psd_vars)))
            JuMP.@constraint(model, coeffs_lhs .== interp_vals .- a)
        else
            for P in Ps
                L = size(P, 2)
                psd_vec = [JuMP.@expression(model, sum(P[u, i] * P[u, j] * μ[u] for u in 1:U)) for i in 1:L for j in 1:i]
                JuMP.@constraint(model, psd_vec in MOI.PositiveSemidefiniteConeTriangle(L))
            end
        end
    end

    return model
end

function test_extra(inst::PolyMinJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    if JuMP.termination_status(model) == MOI.OPTIMAL && !isnan(inst.true_min)
        # check objective value is correct
        tol = eps(T)^0.2
        @test JuMP.objective_value(model) ≈ inst.true_min atol = tol rtol = tol
    end
end

return PolyMinJuMP
