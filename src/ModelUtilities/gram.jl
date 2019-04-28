#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import JuMP
import DynamicPolynomials
const DP = DynamicPolynomials
import Combinatorics
import MathOptInterface
const MOI = MathOptInterface
import Hypatia
const CO = Hypatia.Cones
using LinearAlgebra


# these could become arguments/oracles to generalize
get_lambda(point::Vector{Float64}, P0::Matrix{Float64}) = Symmetric(P0' * Diagonal(point) * P0, :U)
function get_hessian(sqrconstr::JuMP.ConstraintRef, point::Vector{Float64})
    moi_cone = JuMP.constraint_object(sqrconstr).set
    cone = Hypatia.cone_from_moi(moi_cone)
    CO.setup_data(cone)
    cone.point = point
    @assert HYP.Cones.check_in_cone(cone)
    return cone.H
end

# recover Gram matrices, minimized weighted Frobenius distance (Papp and Yildiz)
function gram_frobenius_dist(sqrconstr::JuMP.ConstraintRef, ipwt::Vector{Matrix{Float64}})
    sprimal = JuMP.value(sqrconstr)
    sdual = JuMP.dual(sqrconstr)
    H = get_hessian(sqrconstr, sdual)
    nwts = length(ipwt)
    w = Symmetric(H, :U) \ sprimal
    gram_matrices = Vector{Matrix{Float64}}(undef, nwts)
    for p in 1:nwts
        lambda_inv = inv(Symmetric(get_lambda(sdual, ipwt[p]), :U))
        lambdaw = get_lambda(w, ipwt[p]) #+ 1e-6I
        S = Symmetric(lambda_inv * lambdaw * lambda_inv, :U)
        gram_matrices[p] = S
    end
    return gram_matrices
end

# recover Gram matrices, solving a feasibility problem (could add min logdet objective)
function gram_feasible(sqrconstr::JuMP.ConstraintRef, ipwt::Vector{Matrix{Float64}})
    nwts = length(ipwt)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    sprimal = JuMP.value(sqrconstr)
    U = length(sprimal)
    # TODO proper way to make array of PSD vars?
    G = Vector{Symmetric{JuMP.VariableRef,Array{JuMP.VariableRef,2}}}(undef, nwts)
    for p in eachindex(ipwt)
        L = size(ipwt[p], 2)
        G[p] = JuMP.@variable(model, [1:L, 1:L], Symmetric)
        JuMP.@constraint(model, [G[p][i, j] for i in 1:L for j in 1:i] in MOI.PositiveSemidefiniteConeTriangle(L))
    end
    # JuMP.@constraint(model, [u in 1:U], s[u] == sum(sum(ipwt[p][u, i] * ipwt[p][u, j] * G[p][i, j] for i in 1:size(ipwt[p], 2), j in 1:size(ipwt[p], 2)) for p in 1:nwts)) # TODO collapse
    lhs = sum(diag(ipwt[p] * G[p] * ipwt[p]') for p in 1:nwts)
    JuMP.@constraint(model, sprimal - lhs .== 0)

    JuMP.optimize!(model)
    gram_matrices = Vector{Matrix{Float64}}(undef, nwts)
    for p in eachindex(ipwt)
        gram_matrices[p] = JuMP.value.(G[p])
    end
    return gram_matrices
end
