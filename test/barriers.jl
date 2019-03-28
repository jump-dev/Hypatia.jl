#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Test
using Random
using LinearAlgebra
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
        while !isposdef(lambda(subpoint))
            subpoint .+= ones(cone.U)
        end
    end
    all_lambdas = [Symmetric(lambda(point[((r - 1) * cone.U + 1):(r * cone.U)])) for r in 1:cone.R]
    subpoint = view(cone.point, 1:cone.U)
    while !isposdef(schur_lambda(all_lambdas))
        subpoint .+= rand(cone.U)
        all_lambdas[1] = lambda(subpoint)
    end
    return nothing
end

function test_dependencies(cone::CO.WSOSPolyInterpSOC)
    R = cone.R
    L = size(cone.ipwt[1], 2)
    arrow_mat = kron(Matrix{Float64}(I, R, R), cone.lambda[1][1])
    for r in 2:R
        arrow_mat[((r - 1) * L + 1):(r * L), 1:L] = cone.lambda[1][r]
    end
    arrow_mat_inv = zeros(R * L, R * L)
    for r1 in 1:R, r2 in 1:R
        arrow_mat_inv[((r1 - 1) * L + 1):(r1 * L), ((r2 - 1) * L + 1):(r2 * L)] = CO.mat_inv(cone, r1, r2, 1)
    end
    @test arrow_mat_inv * Symmetric(arrow_mat, :L) ≈ I
    return nothing
end

function pass_through_cone(cone::CO.Cone)
    for _ in 1:100
        load_feasible_point!(cone)
        @test CO.check_in_cone(cone)
        test_dependencies(cone)
        @test -dot(cone.point, cone.g) ≈ CO.get_nu(cone) atol = 1e-9 rtol = 1e-9
        @test Symmetric(cone.H, :U) * cone.point ≈ -cone.g atol = 1e-9 rtol = 1e-9
    end
    return nothing
end

cone = make_default_cone("WSOSPolyInterpSOC", 2, 2, 2)
pass_through_cone(cone)
