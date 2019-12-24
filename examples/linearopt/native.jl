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
    n::Int;
    nzfrac::Float64 = 1.0,
    )
    # generate random data
    @assert 0 < nzfrac <= 1

    # A matrix is sparse iff nzfrac ∈ [0, 1)
    A = (nzfrac >= 1.0) ? rand(T, m, n) : sprand(T, m, n, nzfrac)
    A .*= 10
    b = vec(sum(A, dims = 2))
    c = rand(T, n)
    G = Diagonal(-one(T) * I, n) # TODO uniformscaling
    h = zeros(T, n)
    cones = CO.Cone{T}[CO.Nonnegative{T}(n)]

    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

linearopt1(T::Type{<:Real}) = linearopt(T, 500, 1000)
linearopt2(T::Type{<:Real}) = linearopt(T, 50, 100)
linearopt3(T::Type{<:Real}) = linearopt(T, 15, 20)
linearopt4(T::Type{<:Real}) = linearopt(T, 500, 1000, nzfrac = 0.05)
linearopt5(T::Type{<:Real}) = linearopt(T, 50, 100, nzfrac = 0.15)
linearopt6(T::Type{<:Real}) = linearopt(T, 15, 20, nzfrac = 0.25)

instances_linearopt_all = [
    linearopt1,
    linearopt2,
    linearopt3,
    linearopt4,
    linearopt5,
    linearopt6,
    ]
instances_linearopt_few = [
    linearopt2,
    linearopt3,
    linearopt5,
    linearopt6,
    ]

function test_linearopt(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return
end
