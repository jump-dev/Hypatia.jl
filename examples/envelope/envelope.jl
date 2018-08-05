#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using Alfonso
import MathOptInterface
const MOI = MathOptInterface
using SparseArrays
using LinearAlgebra
using DelimitedFiles
using Random

function build_envelope(npoly, deg, n, d; native=true, use_data=false, rseed=1)
    # TODO allow n > 1
    @assert n == 1

    # generate interpolation
    (L, U, pts, P0, P, w) = Alfonso.cheb2_data(d)
    LWts = fill(binomial(n+d-1, n), n)
    wtVals = 1.0 .- pts.^2
    PWts = [Array((qr(Diagonal(sqrt.(wtVals[:,j]))*P[:,1:LWts[j]])).Q) for j in 1:n]

    # set up MOI problem data
    A = repeat(sparse(1.0I, U, U), outer=(1, npoly))
    b = w
    if use_data
        # use provided data in data folder
        c = vec(readdlm(joinpath(@__DIR__, "data/c$(size(A,2)).txt"), ',', Float64))
    else
        # generate random data
        Random.seed!(rseed)
        LDegs = binomial(n+deg, n)
        c = vec(P[:,1:LDegs]*rand(-9:9, LDegs, npoly))
    end

    cones = Alfonso.ConeData[]
    coneidxs = AbstractUnitRange[]
    for k in 1:npoly
        push!(cones, Alfonso.SumOfSqrData(U, P, PWts))
        push!(coneidxs, 1+(k-1)*U:k*U)
    end

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
# select number of polynomials and degrees for the envelope
# select dimension and SOS degree (to be squared)
# alf =
#     build_envelope(2, 5, 1, 5, use_data=true)
    # build_envelope(2, 5, 1, 5)

# solve it
# @time Alfonso.solve!(alf)
