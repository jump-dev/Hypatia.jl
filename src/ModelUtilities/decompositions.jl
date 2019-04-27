#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import JuMP
import DynamicPolynomials
DP = DynamicPolynomials
import Combinatorics
import MathOptInterface
const MOI = MathOptInterface

function get_decomposition(
        constraint,
        lambda_oracle,
        hessian_oracle,
        n,
        d,
        )
    s = JuMP.value(constraint)
    x = JuMP.dual(constraint) # check sign
    (g, H) = hessian_oracle(x)
    delta = 1e-3
    @show eigen(Symmetric(H, :U)).values
    @show cond(Symmetric(H, :U))
    fH = cholesky(Symmetric(H, :U))
    @show norm(fH.L \ (s + delta * g))

    w = Symmetric(H, :U) \ s
    lambda_inv = inv(Symmetric(lambda_oracle(x), :U))
    @show eigen(lambda_oracle(w)).values
    S = Symmetric(lambda_inv * lambda_oracle(w) * lambda_inv, :U)
    f = cholesky(S)
    basis = build_basis(n, d)
    decomposition = f.U * basis
    return decomposition
end

function get_decomposition(constraint, ipwt::Vector{Matrix{Float64}}, n::Int, d::Int)
    nwts = length(ipwt)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-8))
    s = JuMP.value(constraint)
    U = length(s)
    # TODO proper way to make array of PSD vars?
    G = Vector{Symmetric{JuMP.VariableRef,Array{JuMP.VariableRef,2}}}(undef, nwts)
    for p in eachindex(ipwt)
        L = size(ipwt[p], 2)
        G[p] = JuMP.@variable(model, [1:L, 1:L], Symmetric)
        JuMP.@constraint(model, [G[p][i, j] for i in 1:L for j in 1:i] in MOI.PositiveSemidefiniteConeTriangle(L))
    end
    JuMP.@constraint(model, [u in 1:U], s[u] == sum(sum(ipwt[p][u, i] * ipwt[p][u, j] * G[p][i, j] for i in 1:size(ipwt[p], 2), j in 1:size(ipwt[p], 2)) for p in 1:nwts)) # TODO collapse

    JuMP.optimize!(model)
    for p in eachindex(ipwt)
        Gval = JuMP.value.(G[p])
        @show isposdef(Gval)
        f = cholesky(Symmetric(Gval))
        # basis = build_basis(n, d) # FIXME pass in an array of degrees, used different basis for each
        # @show U, size(ipwt[p])
        # @show f.U, basis
        # @show f.U[1, :] * basis
        # decomposition = sum((f.U * basis).^2)
    end
end

function calc_u(d, monovec)
    n = length(monovec)
    u = Vector{Vector}(undef, n) # TODO type properly
    for j in 1:n
        uj = u[j] = Vector(undef, d+1) # TODO type properly
        uj[1] = DP.Monomial(1)
        uj[2] = monovec[j]
        for t in 3:(d + 1)
            uj[t] = 2.0 * uj[2] * uj[t-1] - uj[t-2]
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
    M = Vector(undef, L)
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
HYP = Hypatia
import JuMP
using PolyJuMP
import DynamicPolynomials
DP = DynamicPolynomials
MU = HYP.ModelUtilities
using LinearAlgebra

n = 2
DP.@polyvar x[1:n]
DP.@polyvar y[1:n]

# numerically unstable
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
get_decomposition(sqrconstr, [P0, PWts...], n, d)

# model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
# cone = HYP.WSOSPolyInterpCone(U, [P0])
# JuMP.@variable(model, dummy[1:U])
# sqrconstr2 = JuMP.@constraint(model, dummy in cone)
# JuMP.optimize!(model)

lambda_oracle(point) = Symmetric(P0' * Diagonal(point) * P0, :U)
function hessian_oracle(point)
    cone = model.moi_backend.optimizer.model.optimizer.cones[1]
    cone.point .= point
    @assert HYP.Cones.check_in_cone(cone)
    H = cone.H
    # if !(isposdef(Symmetric(H, :U)))
    #     H += 1e-3I
    # end
    @show isposdef(H)
    return cone.g, H
end





;
