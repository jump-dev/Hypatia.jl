#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
using DynamicPolynomials # TODO do without
using LinearAlgebra
import Hypatia.Cones

struct RandomPolyMatJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    halfdeg::Int
    R::Int
    formulation::Symbol
end


function build(inst::RandomPolyMatJuMP{T}) where {T <: Float64} # TODO generic reals
    (n, halfdeg, R) = (inst.n, inst.halfdeg, inst.R)
    U = binomial(n + 2 * halfdeg, n)
    L = binomial(n + halfdeg, n)
    svec_dim = div(R * (R + 1), 2)

    free_dom = ModelUtilities.FreeDomain{Float64}(n)
    (U, points, Ps, V) = ModelUtilities.interpolate(free_dom, halfdeg, calc_V = true)

    model = JuMP.Model()
    JuMP.@variable(model, lambda)

    # TODO do without
    @polyvar x[1:n]
    monos = monomials(x, 0:halfdeg)
    half_mat = [dot(rand(L), monos) for _ in 1:R, _ in 1:R]
    full_mat = half_mat * half_mat'
    full_coeffs = [JuMP.AffExpr[full_mat[i, j](points[u, :]...) for u in 1:U] .- (i == j ? lambda : 0) for j in 1:R, i in 1:R]

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
    end
    JuMP.@objective(model, Max, lambda)

    return model
end

# function build(inst::RandomPolyMatJuMP{T}) where {T <: Float64} # TODO generic reals
#     (n, halfdeg, R) = (inst.n, inst.halfdeg, inst.R)
#     U = binomial(n + 2 * halfdeg, n)
#     L = binomial(n + halfdeg, n)
#     svec_dim = div(R * (R + 1), 2)
#
#     free_dom = ModelUtilities.FreeDomain{Float64}(n)
#     (U, points, Ps, V) = ModelUtilities.interpolate(free_dom, halfdeg, calc_V = true)
#
#     model = JuMP.Model()
#     JuMP.@variable(model, lambda[1:(U * svec_dim)])
#     JuMP.@variable(model, t)
#     JuMP.@constraint(model, vcat(t, lambda) in JuMP.SecondOrderCone())
#
#     # TODO do without
#     @polyvar x[1:n]
#     monos = monomials(x, 0:halfdeg)
#     half_mat = [dot(rand(L), monos) for _ in 1:R, _ in 1:R]
#     full_mat = half_mat + half_mat'
#     full_coeffs = [JuMP.AffExpr[full_mat[i, j](points[u, :]...) for u in 1:U] .+ lambda[Cones.block_idxs(U, Cones.svec_idx(i, j))] for j in 1:R, i in 1:R]
#
#     if inst.formulation == :nat_wsos_mat
#         JuMP.@constraint(model, vcat([full_coeffs[i, j] * (i == j ? 1 : sqrt(2)) for j in 1:R for i in 1:j]...) in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(R, U, Ps))
#     elseif inst.formulation == :nat_wsos
#         ypts = zeros(svec_dim, R)
#         idx = 1
#         for j in 1:R
#             for i in 1:(j - 1)
#                 full_coeffs[i, j] .*= 2
#                 full_coeffs[i, j] .+= full_coeffs[i, i] + full_coeffs[j, j]
#                 ypts[idx, i] = ypts[idx, j] = 1
#                 idx += 1
#             end
#             ypts[idx, j] = 1
#             idx += 1
#         end
#         new_Ps = Matrix{Float64}[]
#         for P in Ps
#             push!(new_Ps, kron(ypts, P))
#         end
#         JuMP.@constraint(model, vcat([full_coeffs[i, j] for j in 1:R for i in 1:j]...) in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U * svec_dim, new_Ps))
#     end
#     JuMP.@objective(model, Min, t)
#
#     return model
# end

function test_extra(inst::RandomPolyMatJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    # @show JuMP.value(model[:lambda])
end

example_tests(::Type{RandomPolyMatJuMP{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, verbose = true)
    relaxed_options = (tol_feas = 1e-4, tol_rel_opt = 1e-4, tol_abs_opt = 1e-4)
    return [
    ((2, 2, 15, :nat_wsos_mat), nothing, options),
    ((2, 2, 15, :nat_wsos), nothing, options),
    ]
end

return RandomPolyMatJuMP
