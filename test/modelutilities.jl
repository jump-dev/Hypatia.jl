#=
tests for ModelUtilities module
=#

using Test
import Random
using LinearAlgebra
import Hypatia.ModelUtilities

function test_svec_conversion(T::Type{<:Real})
    tol = 10eps(T)
    rt2 = sqrt(T(2))

    vec = rand(T, 6)
    vec_copy = copy(vec)
    Cones.vec_to_svec!(vec)
    @test vec ≈ vec_copy .* [1, rt2, 1, rt2, rt2, 1] atol=tol rtol=tol

    copyto!(vec, vec_copy)
    Cones.vec_to_svec!(vec, incr = 2)
    @test vec ≈ vec_copy .* [1, 1, rt2, rt2, 1, 1] atol=tol rtol=tol

    mat = rand(T, 10, 3)
    mat_copy = copy(mat)
    Cones.vec_to_svec!(mat)
    @test mat ≈ mat_copy .* [1, rt2, 1, rt2, rt2, 1,
        rt2, rt2, rt2, 1] atol=tol rtol=tol

    mat = rand(T, 12, 3)
    mat_copy = copy(mat)
    Cones.vec_to_svec!(mat, incr = 2)
    @test mat ≈ mat_copy .* [1, 1, rt2, rt2, 1, 1, rt2,
        rt2, rt2, rt2, 1, 1] atol=tol rtol=tol
end

function test_fekete_sample(T::Type{<:Real})
    Random.seed!(1)
    n = 3
    halfdeg = 2
    box = ModelUtilities.Box{T}(-ones(T, n), ones(T, n))
    free = ModelUtilities.FreeDomain{T}(n)

    for sample in (true, false)
        (box_U, box_pts, box_Ps) = ModelUtilities.interpolate(
            box, halfdeg, sample = sample, sample_factor = 20)
        (free_U, free_pts, free_Ps) = ModelUtilities.interpolate(
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
        (U, pts, Ps, V, w) = ModelUtilities.interpolate(
            ModelUtilities.FreeDomain{T}(1), halfdeg,
            sample = false, calc_w = true)

        @test dot(w, [sum(pts[i, 1] ^ d for d in 0:(2halfdeg)) for i in 1:U]) ≈
            sum(2 / (i + 1) for i in 0:2:(2halfdeg))
    end
end
