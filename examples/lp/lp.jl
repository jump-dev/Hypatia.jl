#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple LP min c'x s.t. Ax = b, x >= 0
=#

using Alfonso
import MathOptInterface
const MOI = MathOptInterface
using SparseArrays
using LinearAlgebra
using DelimitedFiles
using Random

function build_lp(m, n; native=true, use_data=false, dense=false, nzfrac=1/sqrt(n), rseed=1)
    if use_data
        # use provided data in data folder
        datapath = joinpath(@__DIR__, "data")
        A = readdlm(joinpath(datapath, "A$(m)x$(n).txt"), ',', Float64)
        b = vec(readdlm(joinpath(datapath, "b$m.txt"), ',', Float64))
        c = vec(readdlm(joinpath(datapath, "c$n.txt"), ',', Float64))
    else
        # generate random data
        Random.seed!(rseed)
        if dense
            A = rand(-9.0:9.0, m, n)
        else
            A = 10.0.*sprandn(m, n, nzfrac)
        end
        b = A*ones(n)
        c = rand(0.0:9.0, n)
    end

    cones = [Alfonso.NonnegData(n),]
    coneidxs = [1:n,]

    if native
        # use native interface
        alf = Alfonso.AlfonsoOpt(maxiter=100, verbose=true)
        Alfonso.load_data!(alf, A, b, c, cones, coneidxs)
        return alf
    else
        error("MOI tests not implemented")
        # opt = Alfonso.Optimizer()
        # return opt
    end
end

# optionally use fixed data in folder
# select the random matrix size, dense/sparse, sparsity fraction
# alf =
    # build_lp(500, 1000)
    # build_lp(500, 1000, use_data=true)

# solve it
# @time Alfonso.solve!(alf)
