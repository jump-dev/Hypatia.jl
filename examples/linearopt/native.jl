#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple linear optimization problem (LP) min c'x s.t. Ax = b, x >= 0
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const LS = HYP.LinearSystems
const MU = HYP.ModelUtilities

using SparseArrays
using DelimitedFiles
import Random
using Test

function build_linearopt(
    m::Int,
    n::Int;
    use_data::Bool = false,
    usedense::Bool = false,
    nzfrac::Float64 = inv(sqrt(n)),
    tosparse::Bool = false,
    rseed::Int = 1,
    )
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
        A = usedense ? rand(-9.0:9.0, m, n) : 10.0 .* sprandn(m, n, nzfrac)
        b = A * ones(n)
        c = rand(0.0:9.0, n)
    end
    if tosparse && !issparse(A)
        A = sparse(A)
    end
    G = Diagonal(-1.0I, n) # TODO uniformscaling
    h = zeros(n)

    cone = CO.Cone([CO.Nonnegative(n)], [1:n])

    return (c, A, b, G, h, cone)
end

function run_linearopt()
    # optionally use fixed data in folder
    # select the random matrix size, dense/sparse, sparsity fraction
    (c, A, b, G, h, cone) =
        # build_linearopt(500, 1000, use_data=true)
        # build_linearopt(500, 1000)
        build_linearopt(15, 20)

    HYP.check_data(c, A, b, G, h, cone)
    (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = HYP.preprocess_data(c, A, b, G, useQR=true)
    L = LS.QRSymm(c1, A1, b1, G1, h, cone, Q2, RiQ1)

    model = HYP.Model(verbose=true)
    HYP.load_data!(model, c1, A1, b1, G1, h, cone, L)
    HYP.solve!(model)

    x = zeros(length(c))
    x[dukeep] = HYP.get_x(model)
    y = zeros(length(b))
    y[prkeep] = HYP.get_y(model)
    s = HYP.get_s(model)
    z = HYP.get_z(model)

    status = HYP.get_status(model)
    solvetime = HYP.get_solve_time(model)
    primal_obj = HYP.get_primal_obj(model)
    dual_obj = HYP.get_dual_obj(model)

    @test status == :Optimal

    return
end
