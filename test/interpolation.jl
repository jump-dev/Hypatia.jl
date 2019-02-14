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

function test_recover_interpolant_polys()
    n = 1
    deg = 1
    pts = Matrix{Float64}(undef, 2, 1)
    pts .= [0; 1]
    interpolant_polys = recover_interpolant_polys(pts, n, deg)

    random_pts = rand(5)
    @test interpolant_polys[1].(random_pts) ≈ 1 .- random_pts
    @test interpolant_polys[2].(random_pts) ≈ random_pts

    deg = 2
    pts = Matrix{Float64}(undef, 3, 1)
    pts .= [0; 1; 2]
    interpolant_polys = MU.recover_interpolant_polys(pts, n, deg)

    random_pts = rand(5)
    @test interpolant_polys[1].(random_pts) ≈ (random_pts .- 1.0) .* (random_pts .- 2.0) * 0.5
    @test interpolant_polys[2].(random_pts) ≈ random_pts .* (random_pts .- 2.0) * (-1.0)
    @test interpolant_polys[3].(random_pts) ≈ random_pts .* (random_pts .- 1.0) * 0.5

    n = 2
    deg = 2
    pts = rand(6, 2)
    interpolant_polys = MU.recover_interpolant_polys(pts, n, deg)

    for i in 1:6, j in 1:6
        if j == i
            @test interpolant_polys[i](pts[j, :]) ≈ 1.0 atol = 1e-9
        else
            @test interpolant_polys[i](pts[j, :]) ≈ 0.0 atol = 1e-9
        end
    end

    for n in 1:3, sample in [true, false]
        d = 2
        (U, pts, P0, PWts, w) = MU.interpolate(MU.FreeDomain(n), d, sample = sample, calc_w = true)
        DynamicPolynomials.@polyvar x[1:n]
        monos = DynamicPolynomials.monomials(x, 0:(2 * d))
        interpolant_polys = MU.recover_interpolant_polys(pts, n, 2 * d)

        @test sum(interpolant_polys) ≈ 1.0
        @test sum(w[i] * interpolant_polys[j](pts[i, :]) for j in 1:U, i in 1:U) ≈ sum(w)
        @test sum(w) ≈ 2^n
    end

end
