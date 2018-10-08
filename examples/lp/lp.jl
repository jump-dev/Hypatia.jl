#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple LP min c'x s.t. Ax = b, x >= 0
=#

using Hypatia
using SparseArrays
using DelimitedFiles
using Random

function build_lp!(
    m::Int,
    n::Int;
    use_data::Bool = false,
    dense::Bool = false,
    nzfrac::Float64 = 1/sqrt(n),
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
    G = Diagonal(-1.0I, n) # TODO uniformscaling
    h = zeros(n)

    cone = Hypatia.Cone([Hypatia.NonnegativeCone(n)], [1:n])

    return (c, A, b, G, h, cone)
end

function run_lp()
    # optionally use fixed data in folder
    # select the random matrix size, dense/sparse, sparsity fraction
    (c, A, b, G, h, cone) =
        # build_lp!(500, 1000, use_data=true)
        # build_lp!(500, 1000)
        build_lp!(15, 20)

    Hypatia.check_data(c, A, b, G, h, cone)
    (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)
    L = Hypatia.QRSymmCache(c1, A1, b1, G1, h, Q2, RiQ1)

    opt = Hypatia.Optimizer(maxiter=100, verbose=false)
    Hypatia.load_data!(opt, c1, A1, b1, G1, h, cone, L)
    Hypatia.solve!(opt)

    x = zeros(length(c))
    x[dukeep] = Hypatia.get_x(opt)
    y = zeros(length(b))
    y[prkeep] = Hypatia.get_y(opt)
    s = Hypatia.get_s(opt)
    z = Hypatia.get_z(opt)

    status = Hypatia.get_status(opt)
    solvetime = Hypatia.get_solvetime(opt)
    pobj = Hypatia.get_pobj(opt)
    dobj = Hypatia.get_dobj(opt)

    # @show status
    # @show x
    # @show pobj
    # @show dobj
    return nothing
end
