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

function envelope_native(
    ::Type{T},
    n::Int,
    rand_halfdeg::Int,
    num_polys::Int,
    env_halfdeg::Int,
    primal_wsos::Bool; # use primal formulation, else use dual
    domain::MU.Domain = MU.Box{T}(-ones(T, n), ones(T, n)),
    sample::Bool = true,
    sample_factor::Int = 100,
    ) where {T <: Real}
    @assert n == MU.get_dimension(domain)
    @assert rand_halfdeg <= env_halfdeg

    # generate interpolation
    (U, pts, Ps, w) = MU.interpolate(domain, env_halfdeg, calc_w = true, sample = sample, sample_factor = sample_factor)

    # generate random data
    L = binomial(n + rand_halfdeg, n)
    c_or_h = vec(Ps[1][:, 1:L] * rand(T(-9):T(9), L, num_polys))

    if primal_wsos
        # WSOS cone in primal
        c = -w
        A = zeros(T, 0, U)
        b = T[]
        G = repeat(sparse(one(T) * I, U, U), outer = (num_polys, 1))
        h = c_or_h
    else
        # WSOS cone in dual
        c = c_or_h
        A = repeat(sparse(one(T) * I, U, U), outer = (1, num_polys))
        b = w
        G = Diagonal(-one(T) * I, num_polys * U) # TODO uniformscaling
        h = zeros(T, num_polys * U)
    end

    cones = CO.Cone{T}[CO.WSOSInterpNonnegative{T, T}(U, Ps, use_dual = !primal_wsos) for k in 1:num_polys]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

function test_envelope_native(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = envelope_native(T, instance...)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return r
end

envelope_native_fast = [
    (2, 2, 3, 4, true),
    (2, 2, 3, 4, false),
    (2, 3, 2, 4, true),
    (2, 3, 2, 4, false),
    (3, 3, 3, 3, true),
    (3, 3, 3, 3, false),
    (3, 3, 5, 4, true),
    (5, 2, 5, 2, true),
    (1, 30, 2, 30, true),
    (1, 30, 2, 30, false),
    (10, 1, 3, 1, true),
    (10, 1, 3, 1, false),
    ]
envelope_native_slow = [
    # TODO below are too slow, need all boolean combinations
    # (3, 3, 5, 4, false),
    # (5, 2, 5, 2, false),
    ]
