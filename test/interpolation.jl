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
