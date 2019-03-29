#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Test
using Random
using LinearAlgebra
using ForwardDiff
using DiffResults
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MU = HYP.ModelUtilities

Random.seed!(1)

load_feasible_point!(::CO.Cone) = error()
test_dependencies(::CO.Cone) = error()
compare_autodiff(::CO.Cone) = error()
function make_default_cone(cone::String, n::Int, d::Int, R::Int)
    if cone == "WSOSPolyInterpSOC"
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), d, sample = false)
        return CO.WSOSPolyInterpSOC(R, U, [P0], true)
    else
        error()
    end
end

function load_feasible_point!(cone::CO.WSOSPolyInterpSOC)
    lambda(point) = Symmetric(cone.ipwt[1]' * Diagonal(point) * cone.ipwt[1])
    schur_lambda(lambdas) = Symmetric(lambdas[1] - sum(lambdas[i] * (lambdas[1] \ lambdas[i]) for i in 2:cone.R))
    point = cone.point
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
    return nothing
end

function test_dependencies(cone::CO.WSOSPolyInterpSOC)
    R = cone.R
    L = size(cone.ipwt[1], 2)
    ipwtj = cone.ipwt[1]
    arrow_mat = kron(Matrix{Float64}(I, R, R), cone.lambda[1][1])
    for r in 2:R
        arrow_mat[((r - 1) * L + 1):(r * L), 1:L] = cone.lambda[1][r]
    end
    arrow_mat_inv = inv(Symmetric(arrow_mat, :L))
    for r in 1:R, r2 in 1:r
        @test cone.PlambdaiP[1][r][r2] ≈ ipwtj * arrow_mat_inv[((r - 1) * L + 1):(r * L), ((r2 - 1) * L + 1):(r2 * L)] * ipwtj'
    end
    return nothing
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
    return nothing
end

function pass_through_cone(cone::CO.Cone)
    for _ in 1:100
        load_feasible_point!(cone)
        @test CO.check_in_cone(cone)
        test_dependencies(cone)
        # compare_autodiff(cone)
        @testset "gradient/hessian" begin
            @test -dot(cone.point, cone.g) ≈ CO.get_nu(cone) atol = 1e-9 rtol = 1e-9
            @test Symmetric(cone.H, :U) * cone.point ≈ -cone.g atol = 1e-9 rtol = 1e-9
        end
    end
    return nothing
end

@testset "poly SOC barrier" begin
    cone = make_default_cone("WSOSPolyInterpSOC", 2, 2, 2)
    pass_through_cone(cone)
end
