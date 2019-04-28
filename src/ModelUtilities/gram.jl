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

# recover Gram matrices, minimizing weighted Frobenius distance
function gram_frobenius_dist(
        sqrconstr::JuMP.ConstraintRef,
        ipwt::Vector{Matrix{Float64}},
        n::Int,
        degs::Vector{Int},
        basisvars::Vector{DynamicPolynomials.PolyVar{true}},
        )
    sprimal = JuMP.value(sqrconstr)
    sdual = JuMP.dual(sqrconstr)
    H = get_hessian(sqrconstr, sdual)
    nwts = length(degs)
    w = Symmetric(H, :U) \ sprimal
    gram_matrices = Vector{Matrix{Float64}}(undef, nwts)
    bases = Vector{Any}(undef, nwts) # TODO type
    for p in 1:nwts
        lambda_inv = inv(Symmetric(get_lambda(sdual, ipwt[p]), :U))
        lambdaw = get_lambda(w, ipwt[p]) #+ 1e-6I
        S = Symmetric(lambda_inv * lambdaw * lambda_inv, :U)
        gram_matrices[p] = S
        bases[p] = build_basis(basisvars, degs[p])
    end
    return (gram_matrices, bases)
end

# recover Gram matrices, solving a feasibility problem (could add min logdet objective)
function gram_feasible(sqrconstr::JuMP.ConstraintRef, ipwt::Vector{Matrix{Float64}}, n::Int, degs::Vector{Int}, basisvars::Vector{DynamicPolynomials.PolyVar{true}})
    nwts = length(ipwt)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-8))
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
    bases = Vector{Any}(undef, nwts)
    for p in eachindex(ipwt)
        gram_matrices[p] = JuMP.value.(G[p])
        bases[p] = build_basis(basisvars, degs[p])
    end
    return (gram_matrices, bases)
end

function calc_u(d, monovec)
    n = length(monovec)
    u = Vector{Vector}(undef, n)
    for j in 1:n
        uj = u[j] = Vector{DP.Polynomial{true,Int64}}(undef, d + 1)
        uj[1] = DP.Monomial(1)
        uj[2] = monovec[j]
        for t in 3:(d + 1)
            uj[t] = 2.0 * uj[2] * uj[t - 1] - uj[t - 2]
        end
    end
    return u
end

# returns the basis dynamic polynomials "to be squared"
function build_basis(x, d)
    n = length(x)
    u = calc_u(d, x)
    L = binomial(n + d, n)
    m = Vector{Float64}(undef, L)
    m[1] = 2^n
    M = Vector{DP.Polynomial{true,Int64}}(undef, L)
    M[1] = DP.Monomial(1)

    col = 1
    for t in 1:d
        for xp in Combinatorics.multiexponents(n, t)
            col += 1
            if any(isodd, xp)
                m[col] = 0.0
            else
                m[col] = m[1] / prod(1.0 - abs2(xp[j]) for j in 1:n)
            end
            M[col] = u[1][xp[1] + 1]
            for j in 2:n
                M[col] *= u[j][xp[j] + 1]
            end
        end
    end
    return M
end
