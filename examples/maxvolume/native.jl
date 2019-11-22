#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find maximum volume hypercube with edges parallel to the axes inside a polyhedron
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
    poly_hrep .+= randn(n, n) * T(10) ^ (-n)
    c = vcat(-1, zeros(T, n))
    A = hcat(zeros(T, n), poly_hrep)
    b = ones(T, n)

    if use_geomean
        G = -Matrix{T}(I, n + 1, n + 1)
        h = zeros(T, n + 1)
        cones = CO.Cone{T}[CO.HypoGeomean{T}(fill(inv(T(n)), n))]
    else
        cones = CO.Cone{T}[]
        # number of 3-dimensional power cones needed is n - 1, number of new variables is n - 2
        len_power = 3 * (n - 1)
        G_geo_orig = zeros(T, len_power, n)
        G_geo_newvars = zeros(T, len_power, n - 2)
        c = vcat(c, zeros(T, n - 2))
        A = hcat(A, zeros(T, n, n - 2))

        # first cone is a special case since two of the original variables participate in it
        G_geo_orig[1, 1] = -1
        G_geo_orig[2, 2] = -1
        G_geo_newvars[3, 1] = -1
        push!(cones, CO.Power{T}(fill(inv(T(2)), 2), 1))
        offset = 4
        # loop over new vars
        for i in 1:(n - 3)
            G_geo_newvars[offset + 2, i + 1] = -1
            G_geo_newvars[offset + 1, i] = -1
            G_geo_orig[offset, i + 2] = -1
            push!(cones, CO.Power{T}([inv(T(i + 2)), T(i + 1) / T(i + 2)], 1))
            offset += 3
        end
        # last row also special becuase hypograph variable is involved
        G_geo_orig[offset, n] = -1
        G_geo_newvars[offset + 1, n - 2] = -1
        G = [
            vcat(zeros(T, len_power - 1), -one(T))  G_geo_orig  G_geo_newvars
            ]
        push!(cones, CO.Power{T}([inv(T(n)), T(n - 1) / T(n)], 1))
        h = zeros(T, 3 * (n - 1))
    end

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

test_maxvolume.(maxvolume_all)
