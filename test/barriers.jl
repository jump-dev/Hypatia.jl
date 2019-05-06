#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

Random.seed!(1234)

function load_feasible_point(cone::CO.Cone)
    point = zeros(CO.dimension(cone))
    CO.set_initial_point(point, cone)
    CO.load_point(cone, point)
    return
end

function load_feasible_point(cone::CO.WSOSPolyInterpSOC)
    lambda(point) = Symmetric(cone.ipwt[1]' * Diagonal(point) * cone.ipwt[1])
    schur_lambda(lambdas) = Symmetric(lambdas[1] - sum(lambdas[i] * (lambdas[1] \ lambdas[i]) for i in 2:cone.R))
    point = zeros(CO.dimension(cone))
    cone.point = point
    for r in 1:cone.R
        subpoint = view(point, ((r - 1) * cone.U + 1):(r * cone.U))
        subpoint .= randn(cone.U)
    end
    # randomly add some near-zeros
    nz = rand(1:cone.U)
    near_zeros = rand(1:cone.U, nz)
    point[near_zeros] .= 1e-5
    # make point feasible
    subpoint = view(cone.point, 1:cone.U)
    while !isposdef(lambda(subpoint))
        subpoint .+= rand(cone.U)
    end
    all_lambdas = [Symmetric(lambda(point[((r - 1) * cone.U + 1):(r * cone.U)])) for r in 1:cone.R]
    while !isposdef(schur_lambda(all_lambdas))
        subpoint .+= rand(cone.U)
        all_lambdas[1] = lambda(subpoint)
    end
    return
end

function pass_through_cone(cone::CO.Cone, num_checks::Int)
    CO.setup_data(cone)
    for _ in 1:num_checks
        load_feasible_point(cone)
        @test CO.check_in_cone(cone)
        @testset "gradient/hessian" begin
            @test -dot(cone.point, cone.g) ≈ CO.get_nu(cone) atol=1e-9 rtol=1e-9
            @test Symmetric(cone.H, :U) * cone.point ≈ -cone.g atol=1e-9 rtol=1e-9
        end
    end
    return
end

function test_epinormeucl_barrier()
    for dim in [2, 3, 5]
        cone = CO.EpiNormEucl(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epinorinf_barrier()
    for dim in [3, 5, 7]
        cone = CO.EpiNormInf(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epinormspectral_barrier()
    for (n, m) in [(1, 3), (2, 4)]
        cone = CO.EpiNormSpectral(n, m)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epiperpower_barrier()
    for alpha in [1.5, 2.5]
        cone = CO.EpiPerPower(alpha)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epipersquare_barrier()
    for dim in [3, 5, 7]
        cone = CO.EpiPerSquare(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epipersumexp_barrier()
    for dim in [3, 5, 7]
        cone = CO.EpiPerSumExp(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_hypogeomean_barrier()
    for dim in [3, 5, 7]
        alpha = rand(dim - 1)
        alpha ./= sum(alpha)
        cone = CO.HypoGeomean(alpha)
        pass_through_cone(cone, 1)
    end
    return
end

function test_hypoperlog_barrier()
    cone = CO.HypoPerLog()
    pass_through_cone(cone, 1)
    return
end

function test_hypoperlogdet_barrier()
    for dim in [3, 5, 8]
        cone = CO.HypoPerLogdet(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_semidefinite_barrier()
    for dim in [1, 3, 6]
        cone = CO.PosSemidef(dim)
        pass_through_cone(cone, 1)
    end
    return
end

function test_wsospolyinterp_2_barrier()
    for n in 1:3, d in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), d, sample = false)
        Ps = [P0]
        gs = [ones(U)]
        cone = CO.WSOSPolyInterp_2(U, Ps, gs, true)
        pass_through_cone(cone, 1)
    end
    return
end

function test_wsospolyinterp_barrier()
    for n in 1:3, d in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), d, sample = false)
        cone = CO.WSOSPolyInterp(U, [P0], true)
        pass_through_cone(cone, 1)
    end
    return
end

function test_wsospolyinterpmat_barrier()
    for n in 1:3, d in 1:3, R in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), d, sample = false)
        cone = CO.WSOSPolyInterpMat(R, U, [P0], true)
        pass_through_cone(cone, 1)
    end
    return
end

function test_wsospolyinterpsoc_barrier()
    for n in 1:2, d in 1:2, R in 3:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), d, sample = false)
        cone = CO.WSOSPolyInterpSOC(R, U, [P0], true)
        pass_through_cone(cone, 10)
    end
    return
end
