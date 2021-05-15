#=
tests for PolyUtils module
=#

using Test
import Random
using LinearAlgebra
import Hypatia.PolyUtils

function test_fekete_sample(T::Type{<:Real})
    Random.seed!(1)
    n = 3
    halfdeg = 2
    box = PolyUtils.BoxDomain{T}(-ones(T, n), ones(T, n))
    free = PolyUtils.FreeDomain{T}(n)

    for sample in (true, false)
        (box_U, box_pts, box_Ps) = PolyUtils.interpolate(
            box, halfdeg, sample = sample, sample_factor = 20)
        (free_U, free_pts, free_Ps) = PolyUtils.interpolate(
            free, halfdeg, sample = sample, sample_factor = 20)

        @test length(free_Ps) == 1
        @test box_U == free_U
        @test size(box_pts) == size(free_pts)
        @test size(box_Ps[1]) == size(free_Ps[1])
        @test norm(box_Ps[1]) ≈ norm(free_Ps[1]) atol=1e-1 rtol=1e-1
    end
end

function test_cheb2_w(T::Type{<:Real})
    for halfdeg in 1:4
        (U, pts, Ps, V, w) = PolyUtils.interpolate(PolyUtils.FreeDomain{T}(1),
            halfdeg, sample = false, calc_w = true)

        @test dot(w, [sum(pts[i, 1] ^ d for d in 0:(2halfdeg)) for i in 1:U]) ≈
            sum(2 / (i + 1) for i in 0:2:(2halfdeg))
    end
end
