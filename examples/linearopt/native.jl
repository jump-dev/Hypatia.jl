#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple linear optimization problem (LP) min c'x s.t. Ax = b, x >= 0
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
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

linearopt1(; use_dense::Bool = true) = (model = build_linearopt(500, 1000, use_data = true, tosparse = !use_dense), true_obj = 2055.807)
linearopt2(; use_dense::Bool = true) = (model = build_linearopt(500, 1000, tosparse = !use_dense), true_obj = NaN)
linearopt3(; use_dense::Bool = true) = (model = build_linearopt(15, 20, tosparse = !use_dense), true_obj = NaN)
linearopt4(; use_dense::Bool = true) = (model = build_linearopt(25, 50, dense = true, tosparse = !use_dense), true_obj = NaN)

function test_linearopt(instance::Function)
    ((c, A, b, G, h, cones, cone_idxs), true_obj) = instance()
    model = MO.PreprocessedLinearModel(c, A, b, G, h, cones, cone_idxs)
    solver = SO.HSDSolver(model, verbose = true)
    SO.solve(solver)
    r = SO.test_certificates(solver, model, atol = 1e-3, rtol = 1e-3)
    @test r.status == :Optimal
    if !isnan(true_obj)
        @test r.primal_obj ≈ true_obj atol = 1e-4 rtol = 1e-4
    end

    return
end

test_linearopts() = test_linearopt.([linearopt1, linearopt2, linearopt3, linearopt4])
