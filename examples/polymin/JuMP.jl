#=
see description in examples/polymin/native.jl
=#

include(joinpath(@__DIR__, "data_real.jl"))

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

function build(inst::PolyMinJuMP{T}) where {T <: Float64}
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
        cone = Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps, !use_primal)
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
                    JuMP.@SDconstraint(model, psd_r >= 0)
                end
                push!(psd_vars, psd_r)
            end
            coeffs_lhs = JuMP.@expression(model, [u in 1:U], sum(sum(Pr[u, k] * Pr[u, l] * psd_r[k, l] * (k == l ? 1 : 2) for k in 1:size(Pr, 2) for l in 1:k) for (Pr, psd_r) in zip(Ps, psd_vars)))
            JuMP.@constraint(model, coeffs_lhs .== interp_vals .- a)
        else
            for Pr in Ps
                Lr = size(Pr, 2)
                psd_r = [JuMP.@expression(model, sum(Pr[u, i] * Pr[u, j] * μ[u] for u in 1:U)) for i in 1:Lr for j in 1:i]
                if Lr == 1
                    # Mosek cannot handle 1x1 PSD constraints
                    JuMP.@constraint(model, psd_r[1] >= 0)
                else
                    JuMP.@constraint(model, psd_r in MOI.PositiveSemidefiniteConeTriangle(Lr))
                end
            end
        end
    end

    return model
end

function test_extra(inst::PolyMinJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    if (stat == MOI.OPTIMAL) && !isnan(inst.true_min)
        # check objective value is correct
        tol = eps(T)^0.1
        @test JuMP.objective_value(model) ≈ inst.true_min atol=tol rtol=tol
    end
end
