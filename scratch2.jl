



using JuMP
using MathOptInterface
MOI = MathOptInterface
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using SumOfSquares
using PolyJuMP
using Test
using Random

const rt2 = sqrt(2)


n = 1
d = 4
@polyvar x[1:n]

# Random.seed!(1)
# poly = sum(randn() * z for z in monomials(x, 0:d))
# @show poly

poly = x[1]^4 + 2x[1]^2 + 4x[1] + 1

dom = Hypatia.FreeDomain(n)

(U, pts, P0, PWts, _) = Hypatia.interpolate(dom, div(d, 2) + 1, sample=false)


L = size(P0, 2)
@show L

t = x[1]
Z = [1 + 0t, t]
for l in 3:L
    push!(Z, 2*t*Z[l-1] - Z[l-2])
end
@show Z

P0big = [differentiate(Z[l], t, 2)(x => pts[u, :]) for u in 1:U, l in 3:L]
@show P0big

@show pts
# interiorU = [sum((pts[u,1]^a)/a for a in 2:2:d) for u in 1:U]
interiorU = abs2.(pts[:])
Hypatia.getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::Hypatia.WSOSConvexPolyInterp) = (@. arr = interiorU; arr)



model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
@constraint(model, [AffExpr(poly(x => pts[u, :])) for u in 1:U] in WSOSConvexPolyInterpCone(n, U, [P0big]))
JuMP.optimize!(model)
@show JuMP.termination_status(model)
@show JuMP.primal_status(model)



model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
@variable(model, pi[u in 1:U])
@objective(model, Max, -sum(pi[u] * poly(x => pts[u, :]) for u in 1:U))
@constraint(model, pi in WSOSConvexPolyInterpCone(n, U, [P0big], true))
JuMP.optimize!(model)
@show JuMP.termination_status(model)
@show JuMP.primal_status(model)



# H = differentiate(poly, x, 2)
#
# model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
# PolyJuMP.setpolymodule!(model, SumOfSquares)
# @constraint(model, H in PSDCone())
# JuMP.optimize!(model)
# @show JuMP.termination_status(model)
# @show JuMP.primal_status(model)
