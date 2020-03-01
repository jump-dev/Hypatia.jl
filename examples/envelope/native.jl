#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the (dual of the) polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

using LinearAlgebra
using SparseArrays
import Random
using Test
import Hypatia
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities

function envelope(
    T::Type{<:Real},
    npoly::Int,
    rand_halfdeg::Int,
    n::Int,
    env_halfdeg::Int,
    primal_wsos::Bool, # use primal formulation, else use dual
    )
    # generate interpolation
    @assert rand_halfdeg <= env_halfdeg
    domain = MU.Box{T}(-ones(T, n), ones(T, n))
    (U, pts, Ps, w) = MU.interpolate(domain, env_halfdeg, sample = false, calc_w = true)

    # generate random data
    L = binomial(n + rand_halfdeg, n)
    c_or_h = vec(Ps[1][:, 1:L] * rand(T(-9):T(9), L, npoly))

    if primal_wsos
        # WSOS cone in primal
        c = -w
        A = zeros(T, 0, U)
        b = T[]
        G = repeat(sparse(one(T) * I, U, U), outer = (npoly, 1))
        h = c_or_h
    else
        # WSOS cone in dual
        c = c_or_h
        A = repeat(sparse(one(T) * I, U, U), outer = (1, npoly))
        b = w
        G = Diagonal(-one(T) * I, npoly * U) # TODO uniformscaling
        h = zeros(T, npoly * U)
    end

    cones = CO.Cone{T}[CO.WSOSInterpNonnegative{T, T}(U, Ps, use_dual = !primal_wsos) for k in 1:npoly]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

function test_envelope(T::Type{<:Real}, instance::Tuple; options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = envelope(T, instance...)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return r
end

instances_envelope_fast = [
    (2, 3, 2, 4, true),
    (2, 3, 2, 4, false),
    (3, 3, 3, 3, true),
    (3, 3, 3, 3, false),
    (2, 30, 1, 30, true),
    (2, 30, 1, 30, false),
    ]
instances_envelope_slow = [
    # TODO add
    ]
