#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import Hypatia
const HYP = Hypatia
import JuMP
using PolyJuMP
import DynamicPolynomials
const DP = DynamicPolynomials
const MU = HYP.ModelUtilities
using Test

n = 2
DP.@polyvar x[1:n]
motzkin = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
d = 3
dom = MU.Box(-ones(n), ones(n))
(U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = true, sample_factor = 100)
ipwts = [P0, PWts...]
cone = HYP.WSOSPolyInterpCone(U, ipwts)
model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-6, tol_abs_opt = 1e-6))
JuMP.@variable(model, a)
JuMP.@constraint(model, sqrconstr, [motzkin(x => pts[j, :]) - a for j in 1:U] in cone)
JuMP.@objective(model, Max, a)
JuMP.optimize!(model)

ipwt = [P0, PWts...]
# gram_matrices = gram_feasible(sqrconstr, ipwt)
gram_matrices = gram_frobenius_dist(sqrconstr, ipwt)

degs = [3; 2; 2]
basisvars = x
bases = [MU.get_chebyshev_polys(x, d) for d in degs]

weight_funs = [1; 1 - x[1]^2; 1 - x[2]^2]
@show sum(bases[p]' * gram_matrices[p] * bases[p] * weight_funs[p] for p in eachindex(ipwts)) # matches motzkin



# # numerically unstable
# n = 2
# DP.@polyvar x[1:n]
# DP.@polyvar y[1:n]
# d = 3
# monos = PolyJuMP.monomials([x; y], 0:d)
# random_poly = JuMP.dot(rand(length(monos)), monos)
# random_poly_sqr = random_poly^2
# dom = MU.Box(-ones(2n), ones(2n))
# (U, pts, P0, Pwts, _) = MU.interpolate(dom, d, sample_factor = 100, sample = true)
# model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-6, tol_abs_opt = 1e-6))
# cone = HYP.WSOSPolyInterpCone(U, [P0])
# sqrconstr = JuMP.@constraint(model, [random_poly_sqr(pts[u, :]) for u in 1:U] in cone)
# JuMP.optimize!(model)
# # get_decomposition(sqrconstr, lambda_oracle, hessian_oracle, 2n, d)
# ipwt = [P0, Pwts...]
# basisvars = [x; y]
# weight_funs = vcat(1, [x -> 1 - x[i]^2 for i in 1:2n])
# degs = [6; 5; 5; 5; 5]
# get_decomposition(sqrconstr, ipwt, 2n, degs, basisvars, weight_funs)
