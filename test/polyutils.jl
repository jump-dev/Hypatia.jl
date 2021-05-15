#=
tests for PolyUtils module
=#

using Test
import Random
using LinearAlgebra
import Hypatia.PolyUtils
import Hypatia.PolyUtils.interpolate

function test_fekete_sample(T::Type{<:Real})
    Random.seed!(1)
    for n in 1:3, halfdeg in 1:2
        free_dom = PolyUtils.FreeDomain{T}(n)
        box_dom = PolyUtils.BoxDomain{T}(-ones(T, n), ones(T, n))
        ball_dom = PolyUtils.BallDomain{T}(ones(T, n), one(T))
        Q = rand(T, n, n)
        Q = Symmetric(Q * Q')
        ellip_dom = PolyUtils.EllipsoidDomain{T}(ones(T, n), Q)

        for sample in (true, false)
            interp_options = (; sample = sample, sample_factor = 10)

            free = interpolate(free_dom, halfdeg; interp_options...)
            box = interpolate(box_dom, halfdeg; interp_options...)
            @test length(free.Ps) == 1
            @test length(box.Ps) == 1 + n
            @test free.U == box.U
            @test size(box.pts) == size(free.pts)
            @test size(box.Ps[1]) == size(free.Ps[1])
            @test norm(box.Ps[1]) ≈ norm(free.Ps[1]) atol=1e-1 rtol=1e-1

            if sample
                ball = interpolate(ball_dom, halfdeg; interp_options...)
                ellip = interpolate(ellip_dom, halfdeg; interp_options...)
                @test length(ball.Ps) == 2
                @test length(ellip.Ps) == 2
                @test free.U == ball.U == ellip.U
                @test size(ball.pts) == size(ellip.pts)
            end
        end
    end
end

function test_cheb2_w(T::Type{<:Real})
    for halfdeg in 1:4
        (U, pts, Ps, V, w) = interpolate(PolyUtils.FreeDomain{T}(1),
            halfdeg, sample = false, calc_w = true)

        @test dot(w, [sum(pts[i, 1] ^ d for d in 0:(2halfdeg)) for i in 1:U]) ≈
            sum(2 / (i + 1) for i in 0:2:(2halfdeg))
    end
end
