#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple LP min c'x s.t. Ax >= b
=#

using Alfonso
import MathOptInterface
const MOI = MathOptInterface
using SparseArrays
using LinearAlgebra
using DelimitedFiles
using Random


m = 500
n = 1000
use_data = false
dense = false
nzfrac = 1/sqrt(n)

if use_data
    A = readdlm(joinpath(pwd(), "data/A$(m)x$(n).txt"), ',', Float64)
    b = vec(readdlm(joinpath(pwd(), "data/b$m.txt"), ',', Float64))
    c = vec(readdlm(joinpath(pwd(), "data/c$n.txt"), ',', Float64))
else
    srand(100)
    if dense
        A = rand(-9.0:9.0, m, n)
    else
        A = 10.0.*sprandn(m, n, nzfrac)
    end
    b = A*ones(n)
    c = rand(0.0:9.0, n)
end

cones = ConeData[NonnegData(n),]
coneidxs = AbstractUnitRange[1:n,]

# load into optimizer and solve
opt = AlfonsoOptimizer(maxiter=100)
Alfonso.loaddata!(opt, A, b, c, cones, coneidxs)
# @show opt
@time MOI.optimize!(opt)
