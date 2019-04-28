#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import JuMP
import DynamicPolynomials
const DP = DynamicPolynomials
import Combinatorics
import MathOptInterface
const MOI = MathOptInterface

function get_decomposition(
        sqrconstr,
        ipwt::Vector{Matrix{Float64}},
        lambda_oracle,
        hessian_oracle,
        n,
        degs,
        )
    sprimal = JuMP.value(sqrconstr)
    sdual = JuMP.dual(sqrconstr) # check sign
    (g, H) = hessian_oracle(sdual)
    delta = 1e-3
    nwts = length(degs)
    # @show eigen(Symmetric(H, :U)).values
    # @show cond(Symmetric(H, :U))
    # fH = cholesky(Symmetric(H, :U))
    # @show norm(fH.L \ (sprimal + delta * g))

    w = Symmetric(H, :U) \ sprimal
    factorizations = Vector{Matrix{Float64}}(undef, nwts)
    gram_matrices = Vector{Matrix{Float64}}(undef, nwts)
    for p in 1:nwts
        lambda_inv = inv(Symmetric(lambda_oracle(sdual, ipwt[p]), :U))
        # @show eigen(lambda_oracle(sdual, ipwt[p])).values
        lambdaw = lambda_oracle(w, ipwt[p]) + 1e-6I
        S = Symmetric(lambda_inv * lambdaw * lambda_inv, :U)
        f = cholesky(S)
        gram_matrices[p] = S
        factorizations[p] = f.U
    end
    basis = build_basis(n, degs[1])
    @show basis' * gram_matrices[1] * basis
    lagrange_polys = MU.recover_lagrange_polys(pts, 6)
    @show first_poly = dot(lagrange_polys, sprimal)


    # decomposition = f.U * basis
    return decomposition
end

function get_decomposition(sqrconstr, ipwt::Vector{Matrix{Float64}}, n::Int, degs::Vector{Int})
    nwts = length(ipwt)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-8))
    sprimal = JuMP.value(sqrconstr)
    U = length(s)
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
    factorizations = Vector{Matrix{Float64}}(undef, nwts)
    decompositions = Vector{Any}(undef, nwts) # TODO type
    for p in eachindex(ipwt)
        Gval = JuMP.value.(G[p])
        @show isposdef(Gval)
        f = cholesky(Symmetric(Gval))
        factorizations[p] = f.U
    end
    bases = Vector{Any}(undef, nwts)
    for p in eachindex(ipwt)
        basis = build_basis(n, degs[p])
        bases[p] = basis
        decompositions[p] = factorizations[p] * basis
        # @show U, size(ipwt[p])
        # @show f.U, basis
        # @show f.U[1, :] * basis
        # decomposition = sum((f.U * basis).^2)
    end

    # check decomposition makes sense
    # sum((decompositions[1][i])^2 for i in 1:10)
    lagrange_polys = MU.recover_lagrange_polys(pts, 6)
    @show first_poly = dot(lagrange_polys, sprimal)
    @show bases[1]' * JuMP.value.(G[1]) * bases[1]

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

# returns the basis dynamic polynomials
function build_basis(n, d)
    DP.@polyvar x[1:n]
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


using Test
import Hypatia
const HYP = Hypatia
import JuMP
using PolyJuMP
import DynamicPolynomials
const DP = DynamicPolynomials
const MU = HYP.ModelUtilities
using LinearAlgebra
import SumOfSquares

n = 2
# DP.@polyvar x[1:n]
# DP.@polyvar y[1:n]
#
# # numerically unstable
# d = 3
# monos = PolyJuMP.monomials([x; y], 0:d)
# random_poly = JuMP.dot(rand(length(monos)), monos)
# random_poly_sqr = random_poly^2
# (U, pts, P0, _, _) = MU.interpolate(MU.FreeDomain(2n), d, sample_factor = 100, sample = true)
# model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-6, tol_abs_opt = 1e-6))
# cone = HYP.WSOSPolyInterpCone(U, [P0])
# sqrconstr = JuMP.@constraint(model, [random_poly_sqr(pts[u, :]) for u in 1:U] in cone)
# JuMP.optimize!(model)
# get_decomposition(sqrconstr, lambda_oracle, hessian_oracle, 2n, d)
# get_decomposition(sqrconstr, [P0, PWts...], 2n, d)

# polymin example
DP.@polyvar x[1:n]
motzkin = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
d = 3
dom = MU.Box(-ones(2), ones(2))
(U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = true, sample_factor = 100)
cone = HYP.WSOSPolyInterpCone(U, [P0, PWts...])
model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-6, tol_abs_opt = 1e-6))
JuMP.@variable(model, a)
JuMP.@constraint(model, sqrconstr, [motzkin(x => pts[j, :]) - a for j in 1:U] in cone)
JuMP.@objective(model, Max, a)
JuMP.optimize!(model)
degs = [3; 2; 2]
ipwt = [P0, PWts...]
get_decomposition(sqrconstr, ipwt, n, degs)

# model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
# cone = HYP.WSOSPolyInterpCone(U, [P0])
# JuMP.@variable(model, dummy[1:U])
# sqrconstr2 = JuMP.@constraint(model, dummy in cone)
# JuMP.optimize!(model)


# should match SumOfSquares
# model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-7, tol_rel_opt = 1e-7, tol_abs_opt = 1e-7))
# JuMP.@variable(model, a)
# JuMP.@objective(model, Max, a)
# bss = MU.get_domain_inequalities(dom, x)
# JuMP.@constraint(model, sosconstr, motzkin >= a, domain = bss)
# JuMP.optimize!(model)
# SumOfSquares.gram_matrices(sosconstr)


lambda_oracle(point, P0) = Symmetric(P0' * Diagonal(point) * P0, :U)
function hessian_oracle(point)
    cone = model.moi_backend.optimizer.model.optimizer.cones[1]
    tmppoint = copy(cone.point)
    cone.point .= point
    @assert HYP.Cones.check_in_cone(cone)
    H = cone.H
    # if !(isposdef(Symmetric(H, :U)))
    #     H += 1e-3I
    # end
    @show isposdef(Symmetric(H, :U))
    cone.point .= tmppoint
    return cone.g, H
end





;
