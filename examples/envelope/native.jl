#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

modified from https://github.com/dpapp-github/alfonso/blob/master/polyEnv.m
formulates and solves the (dual of the) polynomial envelope problem described in the paper:
D. Papp and S. Yildiz. Sum-of-squares optimization without semidefinite programming
available at https://arxiv.org/abs/1712.01792
=#

include(joinpath(@__DIR__, "../common_native.jl"))
using SparseArrays

struct EnvelopeNative{T <: Real} <: ExampleInstanceNative{T}
    n::Int
    rand_halfdeg::Int
    num_polys::Int
    env_halfdeg::Int
    primal_wsos::Bool # use primal formulation, else use dual
end

function build(inst::EnvelopeNative{T}) where {T <: Real}
    (n, num_polys) = (inst.n, inst.num_polys)
    @assert inst.rand_halfdeg <= inst.env_halfdeg
    domain = ModelUtilities.Box{T}(-ones(T, n), ones(T, n))

    # generate interpolation
    (U, pts, Ps, _, w) = ModelUtilities.interpolate(domain, inst.env_halfdeg, calc_w = true)

    # generate random data
    L = binomial(n + inst.rand_halfdeg, n)
    c_or_h = vec(Ps[1][:, 1:L] * rand(T(-9):T(9), L, num_polys))

    if inst.primal_wsos
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

    cones = Cones.Cone{T}[Cones.WSOSInterpNonnegative{T, T}(U, Ps, use_dual = !inst.primal_wsos) for k in 1:num_polys]

    model = Models.Model{T}(c, A, b, G, h, cones)
    return model
end

instances[EnvelopeNative]["minimal"] = [
    ((1, 2, 2, 2, true),),
    ((1, 2, 2, 2, false),),
    ]
instances[EnvelopeNative]["fast"] = [
    ((2, 2, 3, 2, true),),
    ((2, 2, 3, 2, false),),
    ((3, 3, 3, 3, true),),
    ((3, 3, 3, 3, false),),
    ((3, 3, 5, 4, true),),
    ((5, 2, 5, 3, true),),
    ((1, 30, 2, 30, true),),
    ((1, 30, 2, 30, false),),
    ((10, 1, 3, 1, true),),
    ((10, 1, 3, 1, false),),
    ]
instances[EnvelopeNative]["slow"] = [
    ((3, 3, 5, 4, false),),
    ((5, 2, 5, 3, false),),
    ((4, 6, 4, 5, true),),
    ((4, 6, 4, 5, false),),
    ((2, 30, 4, 30, true),),
    ((2, 30, 4, 30, false),),
    ]
