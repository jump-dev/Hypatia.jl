#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Alfonso
import MathOptInterface
const MOI = MathOptInterface
using Printf
using SparseArrays
using LinearAlgebra
using DelimitedFiles









loc = joinpath(pwd(), "data")

pts = vec(readdlm(joinpath(loc, "pts.txt"), ',', Float64))
w = vec(readdlm(joinpath(loc, "w.txt"), ',', Float64))
P = readdlm(joinpath(loc, "P.txt"), ',', Float64)
P0 = readdlm(joinpath(loc, "P0.txt"), ',', Float64)
pts[abs.(pts) .< 1e-10] .= 0.0
w[abs.(w) .< 1e-10] .= 0.0
P[abs.(P) .< 1e-10] .= 0.0
P0[abs.(P0) .< 1e-10] .= 0.0

n = 1
d = 5
l = binomial(n+d, n)
u = binomial(n+2*d, n)
numPolys = 2
degPolys = 5

LWts = fill(n, binomial(n+2*d, n))
bnu = numPolys*(l + sum(LWts)) + 1.0
wtVals = 1.0 .- pts.^2
PWts = [Array((qr(Diagonal(sqrt.(wtVals[:,j]))*P[:,1:LWts[j]])).Q) for j in 1:n]

A = repeat(sparse(1.0I, u, u), outer=(1, numPolys))
b = w
c = vec(readdlm(joinpath(loc, "c.txt"), ',', Float64))
c[abs.(c) .< 1e-10] .= 0.0

cones = ConeData[]
coneidxs = AbstractUnitRange[]
for k in 1:numPolys
    push!(cones, PolyNonnegData(u, P, PWts))
    push!(coneidxs, 1+(k-1)*u:k*u)
end

@show coneidxs

# load into optimizer and solve
opt = AlfonsoOptimizer(maxiter=100)
Alfonso.loaddata!(opt, A, b, c, cones, coneidxs)
# @show opt
@time MOI.optimize!(opt)
