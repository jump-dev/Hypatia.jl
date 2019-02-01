#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

@testset "free domains" begin
    n = 5
    d = 2
    box = Hypatia.Box(-ones(n), ones(n))
    free = Hypatia.FreeDomain(n)
    (box_U, box_pts, box_P0, box_PWts, _) = Hypatia.interpolate(box, d, sample=true, sample_factor=200)
    (free_U, free_pts, free_P0, free_PWts, _) = Hypatia.interpolate(free, d, sample=true, sample_factor=200)
    @test box_U == free_U
    @test size(box_pts) == size(free_pts)
    @test size(box_P0) == size(free_P0)
    @test norm(box_P0) ≈ norm(free_P0) atol=1.0
    @test isempty(free_PWts)

    (box_U, box_pts, box_P0, box_PWts, _) = Hypatia.interpolate(box, d, sample=false)
    (free_U, free_pts, free_P0, free_PWts, _) = Hypatia.interpolate(free, d, sample=false)
    @test box_U == free_U
    @test size(box_pts) == size(free_pts)
    @test norm(box_P0) ≈ norm(free_P0)
    @test isempty(free_PWts)
end

function scaling_mwe(dom::Hypatia.InterpDomain)
    @polyvar x[1:2]
    (U, pts, P0, PWts, _) = Hypatia.interpolate(dom, 3, sample = true, sample_factor = 50)
    wsos_cone = WSOSPolyInterpCone(U, [P0, PWts...])
    model = Model(with_optimizer(Hypatia.Optimizer, verbose = true))
    @variable(model, v, PolyJuMP.Poly(monomials(x, 0:5)))
    @constraint(model, [v(pts[u, :]) for u in 1:U] in wsos_cone)
    return model
end

@testset "domain scaling" begin
    for dom in [
        Hypatia.Box(-0.01 * ones(2), 0.01 * ones(2)),
        Hypatia.Ball([0.0, 0.0], 0.01),
        Hypatia.Ellipsoid([0.0, 0.0], ones(2, 2) * 0.01)
        ]
        model = scaling_mwe(dom)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.OPTIMAL
    end
end
