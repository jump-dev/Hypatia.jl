#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

# TODO test errors for incompatible inputs e.g. dim is too small
=#

import DiffResults
import ForwardDiff

Random.seed!(1234)

test_dependencies(::CO.Cone) = nothing
compare_autodiff(::CO.Cone) = nothing

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

function test_dependencies(cone::CO.WSOSPolyInterpSOC)
    # checks PlambdaiP have been formed correctly
    R = cone.R
    L = size(cone.ipwt[1], 2)
    ipwtj = cone.ipwt[1]
    lambda1 = ipwtj' * Diagonal(cone.point[1:cone.U]) * ipwtj
    arrow_mat = kron(Matrix{Float64}(I, R, R), lambda1)
    for r in 2:R
        arrow_mat[((r - 1) * L + 1):(r * L), 1:L] = lambda1 * cone.li_lambda[1][r - 1]
    end
    arrow_mat_inv = inv(Symmetric(arrow_mat, :L))
    for r in 1:R
        for r2 in 1:(r - 1)
            @test cone.PlambdaiP[1][r][r2] ≈ ipwtj * arrow_mat_inv[((r - 1) * L + 1):(r * L), ((r2 - 1) * L + 1):(r2 * L)] * ipwtj'
        end
        @test Symmetric(cone.PlambdaiP[1][r][r], :U) ≈ ipwtj * arrow_mat_inv[((r - 1) * L + 1):(r * L), ((r - 1) * L + 1):(r * L)] * ipwtj'
    end
    return
end

function compare_autodiff(cone::CO.WSOSPolyInterpSOC)
    function barfun(point)
        ipwtj = cone.ipwt[1]
        L = size(ipwtj, 2)
        mat = similar(point, L, L)
        point_pq = point[1:cone.U]
        first_lambda = ipwtj' * Diagonal(point_pq) * ipwtj
        mat = Symmetric(first_lambda, :U)
        uo = cone.U + 1
        for p in 2:cone.R
            point_pq = point[uo:(uo + cone.U - 1)]
            tmp = Symmetric(ipwtj' * Diagonal(point_pq) * ipwtj)
            mat -= Symmetric(tmp * (Symmetric(first_lambda, :U) \ tmp'))
            uo += cone.U
        end
        return -logdet(Symmetric(mat, :U))
    end
    diffres = DiffResults.HessianResult(cone.point)
    diffres = ForwardDiff.hessian!(diffres, x -> barfun(x), cone.point)
    adg = DiffResults.gradient(diffres)
    adh = DiffResults.hessian(diffres)
    @test adg ≈ cone.g atol = 1e-9 rtol = 1e-9
    @test adh ≈ Symmetric(cone.H) atol = 1e-9 rtol = 1e-9
    return
end

function pass_through_cone(cone::CO.Cone, num_checks::Int)
    for _ in 1:num_checks
        load_feasible_point(cone)
        @test CO.check_in_cone(cone)
        test_dependencies(cone)
        compare_autodiff(cone)
        @testset "gradient/hessian" begin
            @test -dot(cone.point, cone.g) ≈ CO.get_nu(cone) atol = 1e-9 rtol = 1e-9
            @test Symmetric(cone.H, :U) * cone.point ≈ -cone.g atol = 1e-9 rtol = 1e-9
        end
    end
    return
end

function test_epinormeucl_barrier()
    for dim in [2; 3; 5]
        cone = CO.EpiNormEucl(dim)
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epinorinf_barrier()
    for dim in [3; 5; 7]
        cone = CO.EpiNormInf(dim)
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epinormspectral_barrier()
    for (n, m) in [(1, 3); (2, 4)]
        cone = CO.EpiNormSpectral(n, m)
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epiperpower_barrier()
    for alpha in [1.5; 2.5]
        cone = CO.EpiPerPower(alpha)
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epipersquare_barrier()
    for dim in [3; 5; 7]
        cone = CO.EpiPerSquare(dim)
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_epipersumexp_barrier()
    for dim in [3; 5; 7]
        cone = CO.EpiPerSumExp(dim)
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_hypogeomean_barrier()
    for dim in [3; 5; 7]
        alpha = rand(dim - 1)
        alpha ./= sum(alpha)
        cone = CO.HypoGeomean(alpha)
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_hypoperlog_barrier()
    cone = CO.HypoPerLog()
    CO.setup_data(cone)
    pass_through_cone(cone, 1)
    return
end

function test_hypoperlogdet_barrier()
    for dim in [3; 5; 8]
        cone = CO.HypoPerLogdet(dim)
        # TODO error if dim-2 not a triangular number
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_semidefinite_barrier()
    for dim in [1; 3; 6]
        cone = CO.PosSemidef(dim)
        # TODO error if dim not a triangular number
        CO.setup_data(cone)
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
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_wsospolyinterp_barrier()
    for n in 1:3, d in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), d, sample = false)
        cone = CO.WSOSPolyInterp(U, [P0], true)
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_wsospolyinterpmat_barrier()
    for n in 1:3, d in 1:3, R in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), d, sample = false)
        cone = CO.WSOSPolyInterpMat(R, U, [P0], true)
        CO.setup_data(cone)
        pass_through_cone(cone, 1)
    end
    return
end

function test_wsospolyinterpsoc_barrier()
    for n in 1:2, d in 1:2, R in 3:3
        # TODO error if R less than 2
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), d, sample = false)
        cone = CO.WSOSPolyInterpSOC(R, U, [P0], true)
        CO.setup_data(cone)
        pass_through_cone(cone, 10)
    end
    return
end
