#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple linear optimization problem (LP) min c'x s.t. Ax = b, x >= 0
=#

using SparseArrays
using LinearAlgebra
import Random
using Test
import Hypatia
const CO = Hypatia.Cones

function linearopt(
    T::Type{<:Real},
    m::Int,
    n::Int,
    nz_frac::Float64,
    )
    # generate random data
    @assert 0 < nz_frac <= 1

    A = (nz_frac >= 1.0) ? rand(T, m, n) : sprand(T, m, n, nz_frac)
    A .*= 10
    b = vec(sum(A, dims = 2))
    c = rand(T, n)
    G = Diagonal(-one(T) * I, n) # TODO uniformscaling
    h = zeros(T, n)
    cones = CO.Cone{T}[CO.Nonnegative{T}(n)]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

function test_linearopt(T::Type{<:Real}, instance::Tuple; options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = linearopt(T, instance...)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return r
end

instances_linearopt_fast = [
    (15, 20, 1.0),
    (15, 20, 0.25),
    (50, 100, 1.0),
    (50, 100, 0.15),
    ]
instances_linearopt_slow = [
    (500, 1000, 0.05),
    (500, 1000, 1.0),
    ]
