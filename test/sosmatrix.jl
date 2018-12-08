#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
using MathOptInterface
MOI = MathOptInterface
using JuMP
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using PolyJuMP
using Test

n = 1
deg = 6
d = div(deg-2, 2)

domain = Hypatia.Box(-ones(n), ones(n))
(U, pts, P0, PWts, _) = Hypatia.interpolate(domain, d, sample=true, sample_factor=50)
wsos_mat_cone = WSOSPolyInterpMatCone(n, U, [P0, PWts...])

model = Model(with_optimizer(Hypatia.Optimizer))
@polyvar x[1:n]
@variable(model, p, PolyJuMP.Poly(monomials(x, 0:deg)))
dp = [DynamicPolynomials.differentiate(p, x[j]) for j in 1:n]
Hp = [DynamicPolynomials.differentiate(dp[i], x[j]) for i in 1:n, j in 1:n]

# for n=1, below equivalent to
# @constraint(model, [Hp[1,1](pts[u, :]) for u in 1:U][:] in WSOSPolyInterpCone(U, [P0, PWts...]))

# what we want
@constraint(model, [Hp[i,j](pts[u, :]) for i in 1:n, j in 1:n, u in 1:U][:] in wsos_mat_cone)

# hard coded for n=1 example
@constraint(model, [u in 1:U], Hp[1,1](pts[u,1]) == 30 * pts[u,1][1]^4)
# should be feasible, x^6 is a solution

# @objective(model, Min, p[1])

JuMP.optimize!(model)
JuMP.value(p)
