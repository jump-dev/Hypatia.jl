#=
tests for PolyUtils module
=#

using Test
import Random
using LinearAlgebra
import Hypatia.PolyUtils
import Hypatia.PolyUtils.interpolate

function test_interp_domain(T::Type{<:Real})
    Random.seed!(1)
    @testset "domain interpolation n = $n, halfdeg = $halfdeg" for (n, halfdeg) in
        ((1, 5), (2, 2), (3, 1))
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

function test_complex_interp(T::Type{<:Real})
    @testset "complex interp n = $n, halfdeg = $halfdeg, use_qr = $use_qr" for
        (n, halfdeg) in ((1, 3), (2, 2), (3, 1)), use_qr in (false, true)
        gs = [z -> 1 - sum(abs2, z)]
        (pts, Ps) = interpolate(Complex{T}, halfdeg, n, gs, [1], use_qr = use_qr)
        @test length(Ps) == 2
        L = binomial(n + halfdeg, n)
        U = L^2
        @test length(pts) == U
        @test size(Ps[1]) == (U, L)
    end
end

function test_quadrature(T::Type{<:Real})
    # univariate
    @testset "quadrature n = 1, halfdeg = $halfdeg" for halfdeg in 1:4
        free_dom = PolyUtils.FreeDomain{T}(1)
        free = interpolate(free_dom, halfdeg, sample = false, get_quadr = true)
        pts_polys = [sum(free.pts[i, 1] ^ d for d in
            0:(2 * halfdeg)) for i in 1:free.U]
        @test dot(free.w, pts_polys) ≈ sum(2 / (2i + 1) for i in 0:halfdeg)
    end

    # multivariate box
    @testset "quadrature n = $n, halfdeg = $halfdeg" for (n, halfdeg) in
        ((1, 5), (2, 2), (3, 1))
        box_dom = PolyUtils.BoxDomain{T}(fill(-1, n), fill(3, n))
        box = interpolate(box_dom, halfdeg, sample = false, get_quadr = true)
        @test sum(box.w) ≈ 2^n
    end
end
