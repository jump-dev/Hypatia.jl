#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple linear optimization problem (LP) min c'x s.t. Ax = b, x >= 0
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
# const LS = HYP.LinearSystems
const MU = HYP.ModelUtilities

using SparseArrays
using DelimitedFiles
import Random
using Test

function build_linearopt(
    m::Int,
    n::Int;
    use_data::Bool = false,
    dense::Bool = false,
    nzfrac::Float64 = inv(sqrt(n)),
    tosparse::Bool = false,
    rseed::Int = 1,
    )
    Random.seed!(rseed)

    # set up problem data
    if use_data
        # use provided data in data folder
        datapath = joinpath(@__DIR__, "data")
        A = readdlm(joinpath(datapath, "A$(m)x$(n).txt"), ',', Float64)
        b = vec(readdlm(joinpath(datapath, "b$m.txt"), ',', Float64))
        c = vec(readdlm(joinpath(datapath, "c$n.txt"), ',', Float64))
    else
        # generate random data
        A = dense ? rand(-9.0:9.0, m, n) : 10.0 .* sprandn(m, n, nzfrac)
        b = A * ones(n)
        c = rand(0.0:9.0, n)
    end
    if tosparse && !issparse(A)
        A = sparse(A)
    end
    G = Diagonal(-1.0I, n) # TODO uniformscaling
    h = zeros(n)

    cones = [CO.Nonnegative(n)]
    cone_idxs = [1:n]

    return (c, A, b, G, h, cones, cone_idxs)
end

function run_linearopt()
    # optionally use fixed data in folder
    # select the random matrix size, dense/sparse, sparsity fraction
    (c, A, b, G, h, cones, cone_idxs) =
        # build_linearopt(500, 1000, use_data = true)
        # build_linearopt(500, 1000)
        build_linearopt(15, 20)

    model = MO.LinearObjConic(c, A, b, G, h, cones, cone_idxs)
    solver = IP.HSDESolver(model, verbose = true)
    IP.solve(solver)
    @test IP.get_status(solver) == :Optimal

    return
end
