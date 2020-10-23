#=
find a polynomial f such that f² >= Σᵢ gᵢ² or f >= Σᵢ |gᵢ| where gᵢ are arbitrary polynomials, and the volume under f is minimized
=#

using JuMP
import Hypatia
import Hypatia.ModelUtilities
import Hypatia.Cones
using LinearAlgebra

struct PolyNormJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    rand_deg::Int
    num_polys::Int
    env_halfdeg::Int
    formulation::Symbol
    problem::Symbol
end

function build(inst::PolyNormJuMP{T}) where {T <: Float64} # TODO generic reals
    n = inst.n
    @assert inst.rand_deg <= inst.env_halfdeg * 2
    domain = ModelUtilities.Box{T}(-ones(T, n), ones(T, n))
    num_polys = inst.num_polys

    # generate interpolation
    (U, pts, Ps, V, w) = ModelUtilities.interpolate(domain, inst.env_halfdeg, calc_w = true, calc_V = true)

    # generate random polynomials
    rand_U = binomial(n + inst.rand_deg, n)
    # L = binomial(n + inst.rand_deg, n)
    # polys = Ps[1][:, 1:L] * rand(-9:9, L, inst.num_polys)
    polys = V[:, 1:rand_U] * rand(-9:9, rand_U, inst.num_polys)
    # polys = randn(U, inst.num_polys)

    model = JuMP.Model()
    JuMP.@variable(model, fpv[1:U]) # values at Fekete points
    JuMP.@objective(model, Min, dot(fpv, w)) # integral over domain (via quadrature)

    R = num_polys + 1

    if inst.problem == :L1
        if inst.formulation == :nat_wsos
            p_plus = []
            p_minus = []
            for i in 1:inst.num_polys
                p_plus_i = JuMP.@variable(model, [1:U])
                p_minus_i = JuMP.@variable(model, [1:U])
                push!(p_plus, p_plus_i)
                push!(p_minus, p_minus_i)
                JuMP.@constraint(model, polys[:, i] .== p_plus_i - p_minus_i)
                JuMP.@constraint(model, p_plus_i in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
                JuMP.@constraint(model, p_minus_i in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
            end
            JuMP.@constraint(model, fpv - sum(p_plus) - sum(p_minus) in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
        elseif inst.formulation == :nat_wsos_l1
            JuMP.@constraint(model, vcat(fpv, [polys[:, i] for i in 1:inst.num_polys]...) in Hypatia.WSOSInterpEpiNormOneCone{Float64}(R, U, Ps))
        end
    else
        if inst.formulation == :nat_wsos_soc
            JuMP.@constraint(model, vcat(fpv, [polys[:, i] for i in 1:inst.num_polys]...) in Hypatia.WSOSInterpEpiNormEuclCone{Float64}(R, U, Ps))
        elseif inst.formulation == :nat_wsos_mat
            svec_dim = div(R * (R + 1), 2)
            polyvec = Vector{JuMP.AffExpr}(undef, svec_dim * U)
            polyvec[1:U] .= fpv
            idx = 2
            for j in 2:R
                polyvec[Cones.block_idxs(U, idx)] .= polys[:, (j - 1)] * sqrt(2)
                idx += 1
                for i in 2:(j - 1)
                    polyvec[Cones.block_idxs(U, idx)] .= 0
                    idx += 1
                end
                polyvec[Cones.block_idxs(U, idx)] .= fpv
                idx += 1
            end
            JuMP.@constraint(model, polyvec in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(R, U, Ps))
        elseif inst.formulation == :ext
            psd_vars = []
            for (r, Pr) in enumerate(Ps)
                Lr = size(Pr, 2)
                psd_r = JuMP.@variable(model, [1:(Lr * R), 1:(Lr * R)], Symmetric)
                push!(psd_vars, psd_r)
                JuMP.@SDconstraint(model, psd_r >= 0)
            end
            Ls = [size(Pr, 2) for Pr in Ps]
            JuMP.@constraint(model, [u in 1:U], fpv[u] .== sum(sum(Ps[r][u, k] * Ps[r][u, l] * psd_vars[r][(x1 - 1) * Ls[r] + k, (x1 - 1) * Ls[r] + l] for x1 in 1:R for k in 1:Ls[r] for l in 1:Ls[r]) for r in eachindex(Ls)))
            for x1 in 2:R
                # note that psd_vars[r][(x1 - 1) * Ls[r] + k, (x2 - 1) * Ls[r] + l] is not necessarily symmetric
                coeffs_lhs = JuMP.@expression(model, [u in 1:U], sum(sum(Ps[r][u, k] * Ps[r][u, l] * (psd_vars[r][(x1 - 1) * Ls[r] + k, l] + psd_vars[r][(x1 - 1) * Ls[r] + l, k]) for k in 1:Ls[r] for l in 1:Ls[r]) for r in eachindex(Ls)))
                JuMP.@constraint(model, coeffs_lhs .== polys[:, (x1 - 1)])
            end
        end # formulation
    end # problem type

    return model
end

return PolyNormJuMP
