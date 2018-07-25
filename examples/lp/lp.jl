#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple LP min c'x s.t. Ax >= b
=#

using Alfonso
import MathOptInterface
const MOI = MathOptInterface
using Printf
using SparseArrays
using LinearAlgebra
using DelimitedFiles


loc = joinpath(pwd(), "data")

A = readdlm(joinpath(loc, "A.txt"), ',', Float64)
b = vec(readdlm(joinpath(loc, "b.txt"), ',', Float64))
c = vec(readdlm(joinpath(loc, "c.txt"), ',', Float64))
A[abs.(A) .< 1e-10] .= 0.0
b[abs.(b) .< 1e-10] .= 0.0
c[abs.(c) .< 1e-10] .= 0.0
(m, n) = size(A)

cones = ConeData[NonnegData(n),]
coneidxs = AbstractUnitRange[1:n,]

# load into optimizer and solve
opt = AlfonsoOptimizer(maxiter=100)
Alfonso.loaddata!(opt, A, b, c, cones, coneidxs)
# @show opt
@time MOI.optimize!(opt)
