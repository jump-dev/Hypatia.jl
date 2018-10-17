#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the (dual of the) polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Hypatia
using LinearAlgebra
using SparseArrays
using DelimitedFiles
using Random

function build_envelope!(
    npoly::Int,
    deg::Int,
    n::Int,
    d::Int;
    use_data::Bool = false,
    dense::Bool = false,
    rseed::Int = 1,
    )

    # generate interpolation
    @assert deg <= d
    (L, U, pts, P0, P, w) = Hypatia.interpolate(n, d, calc_w=true)
    LWts = fill(binomial(n+d-1, n), n)
    wtVals = 1.0 .- pts.^2
    PWts = [Array((qr(Diagonal(sqrt.(wtVals[:, j])) * P[:, 1:LWts[j]])).Q) for j in 1:n]

    # set up problem data
    if dense
        A = repeat(Array(1.0I, U, U), outer=(1, npoly))
    else
        A = repeat(sparse(1.0I, U, U), outer=(1, npoly))
    end
    G = Diagonal(-1.0I, npoly*U) # TODO uniformscaling
    b = w
    h = zeros(npoly*U)
    if use_data
        # use provided data in data folder
        c = vec(readdlm(joinpath(@__DIR__, "data/c$(size(A,2)).txt"), ',', Float64))
    else
        # generate random data
        Random.seed!(rseed)
        LDegs = binomial(n+deg, n)
        c = vec(P0[:, 1:LDegs]*rand(-9:9, LDegs, npoly))
    end

    cone = Hypatia.Cone([Hypatia.DualSumOfSquaresCone(U, [P, PWts...]) for k in 1:npoly], [1+(k-1)*U:k*U for k in 1:npoly])

    return (c, A, b, G, h, cone)
end

function run_envelope()
    # optionally use fixed data in folder
    # select number of polynomials and degrees for the envelope
    # select dimension and SOS degree (to be squared)
    (c, A, b, G, h, cone) =
        # build_envelope!(2, 5, 1, 5, use_data=true)
        # build_envelope!(2, 5, 2, 8)
        # build_envelope!(3, 5, 3, 5)
        build_envelope!(2, 3, 1, 4, dense=false)

    Hypatia.check_data(c, A, b, G, h, cone)
    (c1, A1, b1, G1, prkeep, dukeep, Q2, RiQ1) = Hypatia.preprocess_data(c, A, b, G, useQR=true)
    L = Hypatia.QRSymmCache(c1, A1, b1, G1, h, cone, Q2, RiQ1)

    mdl = Hypatia.Model(maxiter=100, verbose=false)
    Hypatia.load_data!(mdl, c1, A1, b1, G1, h, cone, L)
    Hypatia.solve!(mdl)

    x = zeros(length(c))
    x[dukeep] = Hypatia.get_x(mdl)
    y = zeros(length(b))
    y[prkeep] = Hypatia.get_y(mdl)
    s = Hypatia.get_s(mdl)
    z = Hypatia.get_z(mdl)

    status = Hypatia.get_status(mdl)
    solvetime = Hypatia.get_solvetime(mdl)
    pobj = Hypatia.get_pobj(mdl)
    dobj = Hypatia.get_dobj(mdl)

    # @show status
    # @show x
    # @show pobj
    # @show dobj
    return nothing
end
