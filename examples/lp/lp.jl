#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple LP min c'x s.t. Ax = b, x >= 0
=#

using Alfonso
using SparseArrays
using DelimitedFiles
using Random

function build_lp!(alf::Alfonso.AlfonsoOpt, m::Int, n::Int; use_data=false, dense=false, nzfrac::Float64=1/sqrt(n), tosparse=false, rseed::Int=1)
    # set up problem data
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

    if tosparse && !issparse(A)
        A = sparse(A)
    end

    cones = Alfonso.ConeData[Alfonso.NonnegData(n),]
    coneidxs = AbstractUnitRange[1:n,]

    return Alfonso.load_data!(alf, A, b, c, cones, coneidxs)
end

alf = Alfonso.AlfonsoOpt(maxiter=100, verbose=true)

# optionally use fixed data in folder
# select the random matrix size, dense/sparse, sparsity fraction
# build_lp!(alf, 500, 1000)
build_lp!(alf, 500, 1000, use_data=true)

# solve it
@time Alfonso.solve!(alf)
