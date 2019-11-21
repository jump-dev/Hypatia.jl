#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find maximum volume hypercube with edges parallel to the axes inside a polyhdedron
=#
using LinearAlgebra
import Random
using Test
import Hypatia
const CO = Hypatia.Cones

function maxvolume(
    T::Type{<:Real},
    n::Int;
    use_geomean::Bool = true,
    )

    poly_hrep = Matrix{T}(I, n, n)
    poly_hrep .+= randn(n, n) * 10 ^ (-n)
    c = vcat(-1, zeros(T, n))
    A = hcat(zeros(T, n), poly_hrep)
    b = ones(T, n)
    G = -Matrix{T}(I, n + 1, n + 1)
    h = zeros(T, n + 1)
    cones = [CO.Hypogeomean(fill(inv(T(n)), n))]


    return (c = c, A = A, b = b, G = G, h = h, cones = cones)
end

maxvolume1(T::Type{<:Real}) = maxvolume(T, 10, use_geomean = true)
maxvolume2(T::Type{<:Real}) = maxvolume(T, 10, use_geomean = false)

maxvolume_all = [
    maxvolume1,
    maxvolume2,
    ]
maxvolume_few = [
    maxvolume1,
    maxvolume2,
    ]

function test_maxvolume(instance::Function; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = instance(T)
    r = Hypatia.Solvers.build_solve_check(d.c, d.A, d.b, d.G, d.h, d.cones; options...)
    @test r.status == :Optimal
    return
end
