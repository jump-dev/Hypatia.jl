#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple linear optimization problem (LP) min c'x s.t. Ax = b, x >= 0
=#

using SparseArrays
using DelimitedFiles
import Random
using Test
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MU = HYP.ModelUtilities

function linearopt(
    m::Int,
    n::Int;
    use_data::Bool = false,
    use_dense::Bool = true,
    nzfrac::Float64 = inv(sqrt(n)),
    )
    # generate random data
    A = use_dense ? rand(-9.0:9.0, m, n) : 10.0 .* sprandn(m, n, nzfrac)
    b = A * ones(n)
    c = rand(0.0:9.0, n)
    G = Diagonal(-1.0I, n) # TODO uniformscaling
    h = zeros(n)
    cones = [CO.Nonnegative(n)]
    cone_idxs = [1:n]
    return (c = c, A = A, b = b, G = G, h = h, cones = cones, cone_idxs = cone_idxs)
end

linearopt1() = linearopt(500, 1000, use_dense = true)
linearopt2() = linearopt(15, 20, use_dense = true)
linearopt3() = linearopt(500, 1000, use_dense = false)
linearopt4() = linearopt(15, 20, use_dense = false)

function test_linearopt(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    model = MO.PreprocessedLinearModel(d.c, d.A, d.b, d.G, d.h, d.cones, d.cone_idxs)
    solver = SO.HSDSolver(model; options...)
    SO.solve(solver)
    r = SO.test_certificates(solver, model, atol = 1e-3, rtol = 1e-3)
    @test r.status == :Optimal
    return
end

test_linearopt(; options...) = test_linearopt.([
    linearopt1,
    linearopt2,
    linearopt3,
    linearopt4,
    ], options = options)
