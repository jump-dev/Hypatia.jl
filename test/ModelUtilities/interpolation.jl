#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

function fekete_sample()
    Random.seed!(1)
    n = 3
    d = 2
    box = MU.Box(-ones(n), ones(n))
    free = MU.FreeDomain(n)

    for sample in (true, false)
        (box_U, box_pts, box_P0, box_PWts, _) = MU.interpolate(box, d, sample = sample, sample_factor = 20)
        (free_U, free_pts, free_P0, free_PWts, _) = MU.interpolate(free, d, sample = sample, sample_factor = 20)
        @test isempty(free_PWts)
        @test box_U == free_U
        @test size(box_pts) == size(free_pts)
        @test size(box_P0) == size(free_P0)
        @test norm(box_P0) ≈ norm(free_P0) atol = 1e-1 rtol = 1e-1
    end
end

function test_recover_lagrange_polys()
    n = 1
    deg = 1
    pts = reshape(Float64[0, 1], 2, 1)
    lagrange_polys = MU.recover_lagrange_polys(pts, deg)

    random_pts = rand(5)
    @test lagrange_polys[1].(random_pts) ≈ 1 .- random_pts
    @test lagrange_polys[2].(random_pts) ≈ random_pts

    deg = 2
    pts = reshape(Float64[0, 1, 2], 3, 1)
    lagrange_polys = MU.recover_lagrange_polys(pts, deg)

    random_pts = rand(5)
    @test lagrange_polys[1].(random_pts) ≈ (random_pts .- 1.0) .* (random_pts .- 2.0) * 0.5
    @test lagrange_polys[2].(random_pts) ≈ random_pts .* (random_pts .- 2.0) * -1.0
    @test lagrange_polys[3].(random_pts) ≈ random_pts .* (random_pts .- 1.0) * 0.5

    n = 2
    deg = 2
    pts = rand(6, 2)
    lagrange_polys = MU.recover_lagrange_polys(pts, deg)

    for i in 1:6, j in 1:6
        if j == i
            @test lagrange_polys[i](pts[j, :]) ≈ 1.0 atol = 1e-9
        else
            @test lagrange_polys[i](pts[j, :]) ≈ 0.0 atol = 1e-9
        end
    end

    # TODO remove dependency on DynamicPolynomials
    # for n in 1:3, sample in [true, false]
    #     d = 2
    #     (U, pts, P0, PWts, w) = MU.interpolate(MU.FreeDomain(n), d, sample = sample, calc_w = true)
    #     DynamicPolynomials.@polyvar x[1:n]
    #     monos = DynamicPolynomials.monomials(x, 0:(2 * d))
    #     lagrange_polys = MU.recover_lagrange_polys(pts, 2 * d)
    #
    #     @test sum(lagrange_polys) ≈ 1.0
    #     @test sum(w[i] * lagrange_polys[j](pts[i, :]) for j in 1:U, i in 1:U) ≈ sum(w)
    #     @test sum(w) ≈ 2^n
    # end
end

function test_get_chebyshev_polys()
    return true
end
