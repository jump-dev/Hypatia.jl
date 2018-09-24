#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Alfonso
using SparseArrays
using LinearAlgebra
using DelimitedFiles
using Random

function build_envelope!(alf::Alfonso.AlfonsoOpt, npoly::Int, deg::Int, n::Int, d::Int; use_data::Bool=false, dense::Bool=false, rseed::Int=1)
    @assert deg <= d

    # generate interpolation
    (L, U, pts, P0, P, w) = Alfonso.interpolate(n, d, calc_w=true)
    LWts = fill(binomial(n+d-1, n), n)
    wtVals = 1.0 .- pts.^2
    PWts = [Array((qr(Diagonal(sqrt.(wtVals[:, j])) * P[:, 1:LWts[j]])).Q) for j in 1:n]

    # set up problem data
    if dense
        A = repeat(Array(1.0I, U, U), outer=(1, npoly))
    else
        A = repeat(sparse(1.0I, U, U), outer=(1, npoly))
    end
    G = SparseMatrixCSC(-1.0I, npoly*U, npoly*U)
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

    cone = Alfonso.Cone([Alfonso.SumOfSquaresCone(U, [P, PWts...]) for k in 1:npoly], [1+(k-1)*U:k*U for k in 1:npoly])

    return Alfonso.load_data!(alf, c, A, b, G, h, cone)
end

# alf = Alfonso.AlfonsoOpt(maxiter=100, verbose=true)

# optionally use fixed data in folder
# select number of polynomials and degrees for the envelope
# select dimension and SOS degree (to be squared)
# build_envelope!(alf, 2, 5, 1, 5, use_data=true)
# build_envelope!(alf, 2, 5, 2, 8)
# build_envelope!(alf, 3, 5, 3, 5)
# build_envelope!(alf, 2, 3, 3, 5, dense=false)

# @time Alfonso.solve!(alf)
