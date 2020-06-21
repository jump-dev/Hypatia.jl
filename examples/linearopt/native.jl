#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/random_lp.m
solves a simple linear optimization problem (LP) min c'x s.t. Ax = b, x >= 0
=#

include(joinpath(@__DIR__, "../common_native.jl"))
using SparseArrays

struct LinearOptNative{T <: Real} <: ExampleInstanceNative{T}
    m::Int
    n::Int
    nz_frac::Float64
end

example_tests(::Type{<:LinearOptNative{<:Real}}, ::MinimalInstances) = [
    ((2, 4, 1.0),),
    ((2, 4, 0.5),),
    ]
example_tests(::Type{LinearOptNative{Float64}}, ::FastInstances) = [
    # ((15, 20, 1.0),),
    ((15, 20, 0.25),),
    # ((50, 100, 1.0),),
    # ((50, 100, 0.15),),
    ]
example_tests(::Type{LinearOptNative{Float64}}, ::SlowInstances) = [
    ((500, 1000, 0.05),),
    ((500, 1000, 1.0),),
    ]

function build(inst::LinearOptNative{T}) where {T <: Real}
    (m, n, nz_frac) = (inst.m, inst.n, inst.nz_frac)
    @assert 0 < nz_frac <= 1

    # generate random data
    A = (nz_frac >= 1) ? rand(T, m, n) : sprand(T, m, n, nz_frac)
    A .*= 10
    b = vec(sum(A, dims = 2))
    c = rand(T, n)
    G = Diagonal(-one(T) * I, n) # TODO uniformscaling
    h = zeros(T, n)
    cones = Cones.Cone{T}[Cones.Nonnegative{T}(n)]

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end

return LinearOptNative
