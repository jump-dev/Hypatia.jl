using JuMP
import Hypatia
import Hypatia.PolyUtils
using LinearAlgebra

n = 2
halfd = 2
k = 3 # number of functions defining nonnegativity domain

# try good interpolation points from [-1, 1]^n:
# dom = PolyUtils.FreeDomain{Float64}(n)
# try interpoation points from box enclosed in the triangle:
lbs = [1.5, 2]
ubs = [2, 2.5]
dom = PolyUtils.BoxDomain{Float64}(lbs, ubs)
sample = false
(U, pts, Ps, V) = PolyUtils.interpolate(dom, halfd, calc_V = true, sample = sample)
subL = binomial(n + halfd - 1, n)

obj = [(pts[i, 1] - 1)^2 + (pts[i, 1] - pts[i, 2])^2 + (pts[i, 2] - 3)^2 for i in 1:size(pts, 1)]
gs = [1 .- (pts[:, 1] .- 1).^2, 1 .- (pts[:, 1] .- pts[:, 2]).^2, 1 .- (pts[:, 2] .- 3).^2]

model = Model(() -> Hypatia.Optimizer(tol_slow=1e-6))
@variable(model, sos[1:U, 1:(k + 1)])
cone = Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}
for i in 1:k
    @constraint(model, sos[:, i] in cone(U, [Ps[1][:, 1:subL]]))
end
wsos = @constraint(model, sos[:, end] in cone(U, [Ps[1]]))

@variable(model, y)
@constraint(model, y .- obj .== sum(sos[:, i] .* gs[i] for i in 1:k) .+ sos[:, end])

@objective(model, Min, y)
optimize!(model)
objective_value(model)

##

mdual = Model(Hypatia.Optimizer)
@variable(mdual, mu[1:U])
dom_cone = Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, [Ps[1][:, 1:subL]], true)
@constraints(mdual, begin
    weighted[i in 1:k], mu .* gs[i] in dom_cone
    mu in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U, [Ps[1]], true)
    sum(mu) == 1
end)
@objective(mdual, Max, dot(mu, obj))
optimize!(mdual)

##
# get some points that are known to be feasible by scaling down and shifting pts
# dummy_points = pts / 5 .+ [1.75, 2.25]'
dummy_points = pts
F = qr!(Array(V'), Val(true))
# init = PolyUtils.initial_wsos_point(F, dummy_points, [-1], [1], halfd, !sample)
init = PolyUtils.initial_wsos_point(F, dummy_points, lbs, ubs, halfd, !sample)

m2 = Model(() -> Hypatia.Optimizer(tol_slow=1e-6))
@variable(m2, sos[1:U])
@variable(m2, y)
@constraint(m2, y .- obj in
    Hypatia.WSOSInterpNonnegativeCone2{Float64, Float64}(U, init, Ps[1], vcat(fill(subL, k), size(Ps[1], 2)), vcat(gs, [ones(U)])))
@objective(m2, Min, y)
optimize!(m2)
