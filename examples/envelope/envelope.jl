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
using SparseArrays
using LinearAlgebra
using DelimitedFiles
using Random


numPolys = 2
degPolys = 5
n = 1
d = 5
use_data = true

(L, U, pts, w, P0, P) = cheb2_data(d)

LWts = fill(binomial(n+d-1, n), n)
wtVals = 1.0 .- pts.^2
PWts = [Array((qr(Diagonal(sqrt.(wtVals[:,j]))*P[:,1:LWts[j]])).Q) for j in 1:n]

A = repeat(sparse(1.0I, U, U), outer=(1, numPolys))
b = w

if use_data
    c = vec(readdlm(joinpath(pwd(), "data/c$(size(A,2)).txt"), ',', Float64))
else
    srand(100)
    LDegs = binomial(n+degPolys, n)
    c = vec(P[:,1:LDegs]*rand(-9:9, LDegs, numPolys))
end

cones = ConeData[]
coneidxs = AbstractUnitRange[]
for k in 1:numPolys
    push!(cones, SumOfSqrData(U, P, PWts))
    push!(coneidxs, 1+(k-1)*U:k*U)
end


# load into optimizer and solve
opt = AlfonsoOptimizer(maxiter=100)
Alfonso.loaddata!(opt, A, b, c, cones, coneidxs)
# @show opt
@time MOI.optimize!(opt)
