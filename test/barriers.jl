#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using LinearAlgebra
import Random
import Hypatia
import Hypatia.HypReal
const CO = Hypatia.Cones
using Test

function test_barrier_oracles(cone::CO.Cone{T}) where {T <: HypReal}
    CO.setup_data(cone)
    dim = CO.dimension(cone)
    point = Vector{T}(undef, dim)
    CO.set_initial_point(point, cone)
    CO.load_point(cone, point)
    CO.reset_data(cone)

    @test cone.point == point
    @test CO.update_feas(cone)

    nu = CO.get_nu(cone)
    grad = CO.update_grad(cone)
    hess = CO.update_hess(cone)
    inv_hess = CO.update_inv_hess(cone)
    CO.update_hess_prod(cone)
    CO.update_inv_hess_prod(cone)

    tol = max(1e-12, sqrt(eps(T)))

    @test dot(point, grad) ≈ -nu atol=tol rtol=tol
    @test hess * point ≈ -grad atol=tol rtol=tol
    @test hess * inv_hess ≈ I atol=tol rtol=tol

    prod = similar(point)
    @test CO.hess_prod!(prod, point, cone) ≈ -grad atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod, grad, cone) ≈ -point atol=tol rtol=tol

    prod = similar(point, dim, dim)
    @test CO.hess_prod!(prod, inv_hess, cone) ≈ I atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod, hess, cone) ≈ I atol=tol rtol=tol

    return
end

function test_orthant_barrier(T::Type{<:HypReal})
    for dim in [1, 3, 5]
        cone = CO.Nonnegative{T}(dim)
        test_barrier_oracles(cone)
        cone = CO.Nonpositive{T}(dim)
        test_barrier_oracles(cone)
    end
    return
end

function test_epinorminf_barrier(T::Type{<:HypReal})
    for dim in [3, 5, 8]
        cone = CO.EpiNormInf{T}(dim)
        test_barrier_oracles(cone)
    end
    return
end

function test_epinormeucl_barrier(T::Type{<:HypReal})
    for dim in [2, 3, 5]
        cone = CO.EpiNormEucl{T}(dim)
        test_barrier_oracles(cone)
    end
    return
end

# function test_epipersquare_barrier(T::Type{<:HypReal})
#     for dim in [3, 5, 8]
#         cone = CO.EpiPerSquare{T}(dim)
#         test_barrier_oracles(cone)
#     end
#     return
# end
#
# function test_epiperpower_barrier(T::Type{<:HypReal})
#     for alpha in [1.5, 2.5]
#         cone = CO.EpiPerPower{T}(alpha)
#         test_barrier_oracles(cone)
#     end
#     return
# end
#
# function test_epipersumexp_barrier(T::Type{<:HypReal})
#     for dim in [3, 5, 8]
#         cone = CO.EpiPerSumExp{T}(dim)
#         test_barrier_oracles(cone)
#     end
#     return
# end
#
# function test_hypogeomean_barrier(T::Type{<:HypReal})
#     Random.seed!(1)
#     for dim in [3, 5, 8]
#         alpha = rand(T, dim - 1)
#         alpha ./= sum(alpha)
#         cone = CO.HypoGeomean{T}(alpha)
#         test_barrier_oracles(cone)
#     end
#     return
# end
#
# function test_hypoperlog_barrier(T::Type{<:HypReal})
#     cone = CO.HypoPerLog{T}()
#     test_barrier_oracles(cone)
#     return
# end
#
# function test_hypopersumlog_barrier(T::Type{<:HypReal})
#     for dim in [3, 5, 8]
#         cone = CO.HypoPerSumLog{T}(dim)
#         test_barrier_oracles(cone)
#     end
#     return
# end
#
# function test_epinormspectral_barrier(T::Type{<:HypReal})
#     for (n, m) in [(1, 3), (2, 4)]
#         cone = CO.EpiNormSpectral{T}(n, m)
#         test_barrier_oracles(cone)
#     end
#     return
# end
#
# function test_semidefinite_barrier(T::Type{<:HypReal})
#     for dim in [1, 3, 6]
#         cone = CO.PosSemidef{T, T}(dim) # real
#         test_barrier_oracles(cone)
#     end
#     for dim in [1, 4, 9]
#         cone = CO.PosSemidef{T, Complex{T}}(dim) # complex
#         test_barrier_oracles(cone)
#     end
#     return
# end
#
# function test_hypoperlogdet_barrier(T::Type{<:HypReal})
#     for dim in [3, 5, 8]
#         cone = CO.HypoPerLogdet{T}(dim)
#         test_barrier_oracles(cone)
#     end
#     return
# end
#
# function test_wsospolyinterp_barrier(T::Type{<:HypReal})
#     Random.seed!(1)
#     for n in 1:3, halfdeg in 1:3
#         (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
#         P0 = convert(Matrix{T}, P0)
#         cone = CO.WSOSPolyInterp{T, T}(U, [P0], true)
#         test_barrier_oracles(cone)
#     end
#     # TODO also test complex case CO.WSOSPolyInterp{T, Complex{T}} - need complex MU interp functions first
#     return
# end
#
# function test_wsospolyinterpmat_barrier(T::Type{<:HypReal})
#     Random.seed!(1)
#     for n in 1:3, halfdeg in 1:3, R in 1:3
#         (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
#         P0 = convert(Matrix{T}, P0)
#         cone = CO.WSOSPolyInterpMat{T}(R, U, [P0], true)
#         test_barrier_oracles(cone)
#     end
#     return
# end
#
# function test_wsospolyinterpsoc_barrier(T::Type{<:HypReal})
#     Random.seed!(1)
#     for n in 1:2, halfdeg in 1:2, R in 3:3
#         (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
#         P0 = convert(Matrix{T}, P0)
#         cone = CO.WSOSPolyInterpSOC{T}(R, U, [P0], true)
#         test_barrier_oracles(cone)
#     end
#     return
# end
