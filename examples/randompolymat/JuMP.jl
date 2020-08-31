#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

given a random symmetric polynomial matrix P(x), find a polynomial matrix Q(x) with coefficients that have minimum norm so that P(x) + Q(x) are matrix SOS

min t
s.t. t >= norm(Q)
     P + Q matrix-SOS
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
using LinearAlgebra

struct RandomPolyMatJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    halfdeg::Int
    R::Int
    # use_wsosinterppossemideftri::Bool
    # use_wsosinterpnonnegative::Bool
    formulation::Symbol
end

# minimum norm coeffs version
function build(inst::RandomPolyMatJuMP{T}) where {T <: Float64} # TODO generic reals
    (n, halfdeg, R) = (inst.n, inst.halfdeg, inst.R)
    U = binomial(n + 2 * halfdeg, n)
    L = binomial(n + halfdeg, n)
    svec_dim = div(R * (R + 1), 2)

    free_dom = ModelUtilities.FreeDomain{Float64}(n)
    (U, points, Ps) = ModelUtilities.interpolate(free_dom, halfdeg)

    model = JuMP.Model()
    JuMP.@variable(model, q_poly[1:(U * svec_dim)])
    JuMP.@variable(model, t)
    JuMP.@constraint(model, vcat(t, q_poly) in JuMP.SecondOrderCone())
    # JuMP.@constraint(model, vcat(t, q_poly) in JuMP.MOI.NormOneCone(U + 1))
    # JuMP.@constraint(model, vcat(t, q_poly) in JuMP.MOI.NormInfinityCone(U + 1))

    full_mat = [randn(U) for _ in 1:R, _ in 1:R]
    for j in 1:R, i in 1:j
        full_mat[i, j] += full_mat[j, i]
    end
    full_coeffs = full_mat + [q_poly[Cones.block_idxs(U, Cones.svec_idx(i, j))] for j in 1:R, i in 1:R]

    if inst.formulation == :nat_wsos_mat
        JuMP.@constraint(model, vcat([full_coeffs[i, j] * (i == j ? 1 : sqrt(2)) for j in 1:R for i in 1:j]...) in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(R, U, Ps))
    elseif inst.formulation == :nat_wsos
        ypts = zeros(svec_dim, R)
        idx = 1
        for j in 1:R
            for i in 1:(j - 1)
                full_coeffs[i, j] .*= 2
                full_coeffs[i, j] .+= full_coeffs[i, i] + full_coeffs[j, j]
                ypts[idx, i] = ypts[idx, j] = 1
                idx += 1
            end
            ypts[idx, j] = 1
            idx += 1
        end
        new_Ps = Matrix{Float64}[]
        for P in Ps
            push!(new_Ps, kron(ypts, P))
        end
        JuMP.@constraint(model, vcat([full_coeffs[i, j] for j in 1:R for i in 1:j]...) in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U * svec_dim, new_Ps))
    elseif inst.formulation == :ext
        JuMP.@variable(model, psd_var[1:(L * R), 1:(L * R)], PSD)
        for x2 in 1:R, x1 in 1:x2
            coeffs_lhs = JuMP.@expression(model, [u in 1:U], sum(Ps[1][u, k] * Ps[1][u, l] * psd_var[(x1 - 1) * L + k, (x2 - 1) * L + l] for k in 1:L for l in 1:L))
            JuMP.@constraint(model, coeffs_lhs .== full_coeffs[x1, x2])
        end
    else
        error()
    end
    JuMP.@objective(model, Min, t)

    return model
end

# instances[RandomPolyMatJuMP]["minimal"] = [
#     ((1, 1, 2, true, false),),
#     ((1, 1, 2, false, true),),
#     ((1, 1, 2, false, false),),
#     ]
# instances[RandomPolyMatJuMP]["fast"] = [
#     ((3, 2, 4, true, false),),
#     ((3, 2, 4, false, true),),
#     ((3, 2, 4, false, false),),
#     ]
# instances[RandomPolyMatJuMP]["slow"] = [
#     ((4, 2, 10, true, false),),
#     ((4, 2, 10, false, true),),
#     ((4, 2, 10, false, false),),
#     ((2, 2, 15, true, false),),
#     ((2, 2, 15, false, true),),
#     ((2, 2, 15, false, false),),
#     ]

return RandomPolyMatJuMP
