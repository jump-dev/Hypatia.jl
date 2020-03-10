#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple linear optimization problem (LP) min c'x s.t. Ax = b, x >= 0
=#

include(joinpath(@__DIR__, "../common_native.jl"))
using SparseArrays

function linearopt_native(
    ::Type{T},
    m::Int,
    n::Int,
    nz_frac::Float64,
    ) where {T <: Real}
    # generate random data
    @assert 0 < nz_frac <= 1

    A = (nz_frac >= 1.0) ? rand(T, m, n) : sprand(T, m, n, nz_frac)
    A .*= 10
    b = vec(sum(A, dims = 2))
    c = rand(T, n)
    G = Diagonal(-one(T) * I, n) # TODO uniformscaling
    h = zeros(T, n)
    cones = CO.Cone{T}[CO.Nonnegative{T}(n)]

    model = Models.Model{T}(c, A, b, G, h, cones)
    return (model, ())
end

function test_linearopt_native(result, test_helpers, test_options)
    @test result.status == :Optimal
end

options = ()
linearopt_native_fast = [
    ((Float64, 15, 20, 1.0), (), options),
    ((Float64, 15, 20, 0.25), (), options),
    ((Float64, 50, 100, 1.0), (), options),
    ((Float64, 50, 100, 0.15), (), options),
    ]
linearopt_native_slow = [
    ((Float64, 500, 1000, 0.05), (), options),
    ((Float64, 500, 1000, 1.0), (), options),
    ]

@testset "linearopt_native" begin test_native_instance.(linearopt_native, test_linearopt_native, linearopt_native_fast) end
;
