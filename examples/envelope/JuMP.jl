#=
Copyright 2018, Chris Coey and contributors

see description in examples/envelope/native.jl
=#
using JuMP
import Hypatia
import Hypatia.ModelUtilities
import Hypatia.Cones
using LinearAlgebra

struct EnvelopeJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    rand_halfdeg::Int
    num_polys::Int
    env_halfdeg::Int
    formulation::Symbol
end

function build(inst::EnvelopeJuMP{T}) where {T <: Float64} # TODO generic reals
    n = inst.n
    @assert inst.rand_halfdeg <= inst.env_halfdeg
    domain = ModelUtilities.Box{T}(-ones(T, n), ones(T, n))
    num_polys = inst.num_polys

    # generate interpolation
    (U, pts, Ps, V, w) = ModelUtilities.interpolate(domain, inst.env_halfdeg, calc_w = true, calc_V = true)

    # generate random polynomials
    L = binomial(n + inst.rand_halfdeg, n)
    polys = Ps[1][:, 1:L] * rand(-9:9, L, inst.num_polys)
    # polys = V[:, 1:U] * rand(-9:9, U, inst.num_polys)
    # polys = randn(U, inst.num_polys)

    model = JuMP.Model()
    JuMP.@variable(model, fpv[1:U]) # values at Fekete points
    JuMP.@objective(model, Min, dot(fpv, w)) # integral over domain (via quadrature)
    # JuMP.@constraint(model, [i in 1:inst.num_polys], polys[:, i] .- fpv in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, Ps))
    R = num_polys + 1
    if inst.formulation == :nat_wsos_soc
        JuMP.@constraint(model, vcat(fpv, [polys[:, i] for i in 1:inst.num_polys]...) in Hypatia.WSOSInterpEpiNormEuclCone{Float64}(R, U, Ps))
    elseif inst.formulation == :nat_wsos # broken
        # keep in mind this is not equivalent
        svec_dim = div(R * (R + 1), 2)
        ypts = zeros(svec_dim, R)
        polyvec = Vector{JuMP.AffExpr}(undef, svec_dim * U)
        # 11
        polyvec[1:U] .= fpv
        ypts[1, 1] = 1
        idx = 2
        for i in 2:R
            # i1
            polyvec[Cones.block_idxs(U, idx)] .= (2 * polys[:, i - 1] / sqrt(2) + 2 * fpv)
            ypts[idx, i] = ypts[idx, 1] = 1
            idx += 1
            # ij
            for j in 2:(i - 1)
                polyvec[Cones.block_idxs(U, idx)] .= 0
                ypts[idx, i] = ypts[idx, j] = 1
                idx += 1
            end
            polyvec[Cones.block_idxs(U, idx)] .= fpv
            ypts[idx, i] = 1
            idx += 1
        end
        new_Ps = Matrix{Float64}[]
        for P in Ps
            push!(new_Ps, kron(ypts, P))
        end
        JuMP.@constraint(model, polyvec in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(svec_dim * U, new_Ps))
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
    end

    return model
end

# model = build(inst)
# JuMP.set_optimizer(model, Hypatia.Optimizer)
# JuMP.optimize!(model)
# @show JuMP.value.(model[:fpv])
# @show sum(abs, 1:inst.num_polys)
# @show sqrt(sum(abs2, 1:inst.num_polys))


# instances[EnvelopeJuMP]["minimal"] = [
#     ((1, 2, 2, 2, false),),
#     ((1, 2, 2, 2, true),),
#     ]
# instances[EnvelopeJuMP]["fast"] = [
#     ((2, 2, 3, 2, false),),
#     ((3, 3, 3, 3, false),),
#     ((3, 3, 5, 4, false),),
#     ((5, 2, 5, 3, false),),
#     ((1, 30, 2, 30, false),),
#     ((10, 1, 3, 1, false),),
#     ((2, 2, 3, 2, true),),
#     ((3, 3, 3, 3, true),),
#     ((3, 3, 5, 4, true),),
#     ((5, 2, 5, 3, true),),
#     ((1, 30, 2, 30, true),),
#     ((10, 1, 3, 1, true),),
#     ]
# instances[EnvelopeJuMP]["slow"] = [
#     ((4, 5, 4, 6, false),),
#     ((2, 30, 4, 30, false),),
#     ((4, 5, 4, 6, true),),
#     ((2, 30, 4, 30, true),),
#     ]

return EnvelopeJuMP
