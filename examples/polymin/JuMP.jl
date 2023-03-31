#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
see description in examples/polymin/native.jl
=#

include(joinpath(@__DIR__, "data_real.jl"))
include(joinpath(@__DIR__, "data_complex.jl"))

struct PolyMinJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    is_complex::Bool
    interp_vals::Vector
    Ps::Vector{<:Matrix}
    true_min::Real
    use_primal::Bool # solve primal, else solve dual
    use_wsos::Bool # use wsosinterpnonnegative cone, else PSD formulation
end

function PolyMinJuMP{Float64}(is_complex::Bool, poly_name::Symbol, halfdeg::Int, args...)
    R = (is_complex ? Complex{Float64} : Float64)
    interp = get_interp_data(R, poly_name, halfdeg)
    return PolyMinJuMP{Float64}(is_complex, interp..., args...)
end

function PolyMinJuMP{Float64}(is_complex::Bool, n::Int, halfdeg::Int, args...)
    interp = random_interp_data(Float64, n, halfdeg)
    return PolyMinJuMP{Float64}(is_complex, interp..., args...)
end

function build(inst::PolyMinJuMP{T}) where {T <: Float64}
    if !inst.use_wsos && inst.use_primal && inst.is_complex
        error("complex primal PSD formulation is not implemented yet")
    end

    (interp_vals, Ps, use_primal) = (inst.interp_vals, inst.Ps, inst.use_primal)
    U = length(interp_vals)

    model = JuMP.Model()
    if use_primal
        JuMP.@variable(model, a)
        JuMP.@objective(model, Max, a)
    else
        JuMP.@variable(model, μ[1:U])
        JuMP.@objective(model, Min, dot(μ, interp_vals))
        JuMP.@constraint(model, sum(μ) == 1)
    end

    if inst.use_wsos
        R = (inst.is_complex ? Complex{Float64} : Float64)
        cone = Hypatia.WSOSInterpNonnegativeCone{Float64, R}(U, Ps, !use_primal)
        aff_expr = (use_primal ? interp_vals .- a : μ)
        JuMP.@constraint(model, aff_expr in cone)
    else
        if use_primal
            psd_vars = []
            for Pr in Ps
                Lr = size(Pr, 2)
                psd_r = JuMP.@variable(model, [1:Lr, 1:Lr], Symmetric)
                if Lr == 1
                    # Mosek cannot handle 1x1 PSD constraints
                    JuMP.@constraint(model, psd_r[1, 1] >= 0)
                else
                    JuMP.@constraint(model, psd_r in JuMP.PSDCone())
                end
                push!(psd_vars, psd_r)
            end
            coeffs_lhs = JuMP.@expression(
                model,
                [u in 1:U],
                sum(
                    sum(
                        Pr[u, k] * Pr[u, l] * psd_r[k, l] * (k == l ? 1 : 2) for
                        k in 1:size(Pr, 2) for l in 1:k
                    ) for (Pr, psd_r) in zip(Ps, psd_vars)
                )
            )
            JuMP.@constraint(model, coeffs_lhs .== interp_vals .- a)
        else
            psd_set = (inst.is_complex ? JuMP.HermitianPSDCone() : JuMP.PSDCone())
            for Pr in Ps
                Lr = size(Pr, 2)
                psd_r = [
                    JuMP.@expression(model, sum(Pr[u, i]' * Pr[u, j] * μ[u] for u in 1:U)) for i in 1:Lr, j in 1:Lr
                ]
                if Lr == 1
                    # Mosek cannot handle 1x1 PSD constraints
                    JuMP.@constraint(model, real(psd_r[1, 1]) >= 0)
                else
                    JuMP.@constraint(model, Hermitian(psd_r) in psd_set)
                end
            end
        end
    end

    return model
end

function test_extra(inst::PolyMinJuMP{T}, model::JuMP.Model) where {T}
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    if (stat == MOI.OPTIMAL) && !isnan(inst.true_min)
        # check objective value is correct
        tol = eps(T)^0.1
        @test JuMP.objective_value(model) ≈ inst.true_min atol = tol rtol = tol
    end
end
