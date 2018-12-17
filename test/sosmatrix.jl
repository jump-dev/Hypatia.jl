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
using LinearAlgebra

n = 2
deg = 4
d = div(deg-2, 2)

domain = Hypatia.Ball(zeros(n), 1.0)
(U, pts, P0, PWts, _) = Hypatia.interpolate(domain, d, sample=true, sample_factor=50)
wsos_mat_cone = WSOSPolyInterpMatCone(n, U, [P0, PWts...])

model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
@polyvar x[1:n]
@variable(model, p, PolyJuMP.Poly(monomials(x, 0:deg)))
# @variable(model, z)
dp = [DynamicPolynomials.differentiate(p, x[j]) for j in 1:n]
Hp = [DynamicPolynomials.differentiate(dp[i], x[j]) for i in 1:n, j in 1:n]

# for n=1, below equivalent to
# @constraint(model, [Hp[1,1](pts[u, :]) for u in 1:U][:] in WSOSPolyInterpCone(U, [P0, PWts...]))

# what we want
trimat_len(n) = div(n*(n+1),2)
@constraint(model, [Hp[i,j](pts[ui, :]) for j in 1:n, i in 1:n, ui in 1:U if i >= j] in wsos_mat_cone)

# working with monomials, fix some function we like
# @constraint(model, [u in 1:U, i in 1:n, j in 1:n], Hp[i,j](pts[u,:]) == 30 * pts[u,i]^4 * Float64(i == j))
@constraint(model, [u in 1:U, i in 1:n, j in 1:n], Hp[i,j](pts[u,:]) == 12 * pts[u,i]^2 * Float64(i == j))
# should be feasible, sum_i(x_1 ^6) is a solution

# @constraint(model, z >=  p([1.0]))
# @constraint(model, z >=  -p([1.0]))

# @objective(model, Min, z)

JuMP.optimize!(model)
JuMP.value(p)
JuMP.value.(Hp)
