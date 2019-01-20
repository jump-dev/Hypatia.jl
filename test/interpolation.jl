#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

function interp_sample()
    n = 3
    d = 2
    box = MU.Box(-ones(n), ones(n))
    free = MU.FreeDomain(n)
    (box_U, box_pts, box_P0, box_PWts, _) = MU.interpolate(box, d, sample=true, sample_factor=200)
    (free_U, free_pts, free_P0, free_PWts, _) = MU.interpolate(free, d, sample=true, sample_factor=200)
    @test box_U == free_U
    @test size(box_pts) == size(free_pts)
    @test size(box_P0) == size(free_P0)
    @test norm(box_P0) ≈ norm(free_P0) atol=1.0
    @test isempty(free_PWts)
end

function interp_nosample()
    n = 3
    d = 2
    box = MU.Box(-ones(n), ones(n))
    free = MU.FreeDomain(n)
    (box_U, box_pts, box_P0, box_PWts, _) = MU.interpolate(box, d, sample=false)
    (free_U, free_pts, free_P0, free_PWts, _) = MU.interpolate(free, d, sample=false)
    @test box_U == free_U
    @test size(box_pts) == size(free_pts)
    @test norm(box_P0) ≈ norm(free_P0)
    @test isempty(free_PWts)
end
