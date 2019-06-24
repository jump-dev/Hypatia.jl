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
    point = Vector{T}(undef, CO.dimension(cone))
    CO.set_initial_point(point, cone)
    CO.load_point(cone, point)

    @test cone.point == point
    @test CO.check_in_cone(cone)

    tol = max(1e-12, sqrt(eps(T)))
    @test -dot(point, CO.grad(cone)) ≈ CO.get_nu(cone) atol=tol rtol=tol
    @test CO.hess(cone) * point ≈ -CO.grad(cone) atol=tol rtol=tol
    @test CO.hess(cone) * CO.inv_hess(cone) ≈ I atol=tol rtol=tol

    # product with matrices
    for d in [0, 2, 4], _ in 1:5
        prod = zeros(T, cone.dim, cone.dim + d)
        arr = convert(AbstractMatrix{T}, randn(cone.dim, cone.dim + d))
        @test CO.hess_prod!(prod, arr, cone) ≈ CO.hess(cone) * arr atol=tol rtol=tol
        @test CO.inv_hess_prod!(prod, arr, cone) ≈ CO.inv_hess(cone) * arr atol=tol rtol=tol
    end

    # product with vectors
    prod = zeros(T, cone.dim)
    for _ in 1:5
        arr = convert(AbstractVector{T}, randn(cone.dim))
        @test CO.hess_prod!(prod, arr, cone) ≈ CO.hess(cone) * arr atol=tol rtol=tol
        @test CO.inv_hess_prod!(prod, arr, cone) ≈ CO.inv_hess(cone) * arr atol=tol rtol=tol
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

function test_epinorminf_barrier(T::Type{<:HypReal})
    for dim in [3, 5, 8]
        cone = CO.EpiNormInf{T}(dim)
        test_barrier_oracles(cone)
    end
    return
end

function test_epinormspectral_barrier(T::Type{<:HypReal})
    for (n, m) in [(1, 3), (2, 4)]
        cone = CO.EpiNormSpectral{T}(n, m)
        test_barrier_oracles(cone)
    end
    return
end

function test_epiperpower_barrier(T::Type{<:HypReal})
    for alpha in [1.5, 2.5]
        cone = CO.EpiPerPower{T}(alpha)
        test_barrier_oracles(cone)
    end
    return
end

function test_epipersquare_barrier(T::Type{<:HypReal})
    for dim in [3, 5, 8]
        cone = CO.EpiPerSquare{T}(dim)
        test_barrier_oracles(cone)
    end
    return
end

function test_epipersumexp_barrier(T::Type{<:HypReal})
    for dim in [3, 5, 8]
        cone = CO.EpiPerSumExp{T}(dim)
        test_barrier_oracles(cone)
    end
    return
end

function test_hypogeomean_barrier(T::Type{<:HypReal})
    Random.seed!(1)
    for dim in [3, 5, 8]
        alpha = rand(T, dim - 1)
        alpha ./= sum(alpha)
        cone = CO.HypoGeomean{T}(alpha)
        test_barrier_oracles(cone)
    end
    return
end

function test_hypoperlog_barrier(T::Type{<:HypReal})
    cone = CO.HypoPerLog{T}()
    test_barrier_oracles(cone)
    return
end

function test_hypoperlogdet_barrier(T::Type{<:HypReal})
    for dim in [3, 5, 8]
        cone = CO.HypoPerLogdet{T}(dim)
        test_barrier_oracles(cone)
    end
    return
end

function test_hypopersumlog_barrier(T::Type{<:HypReal})
    for dim in [3, 5, 8]
        cone = CO.HypoPerSumLog{T}(dim)
        test_barrier_oracles(cone)
    end
    return
end

function test_semidefinite_barrier(T::Type{<:HypReal})
    for dim in [1, 3, 6]
        cone = CO.PosSemidef{T, T}(dim) # real
        test_barrier_oracles(cone)
    end
    for dim in [1, 4, 9]
        cone = CO.PosSemidef{T, Complex{T}}(dim) # complex
        test_barrier_oracles(cone)
    end
    return
end

function test_wsospolyinterp_barrier(T::Type{<:HypReal})
    Random.seed!(1)
    for n in 1:3, halfdeg in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        P0 = convert(Matrix{T}, P0)
        cone = CO.WSOSPolyInterp{T, T}(U, [P0], true)
        test_barrier_oracles(cone)
    end
    # TODO also test complex case CO.WSOSPolyInterp{T, Complex{T}} - need complex MU interp functions first
    return
end

function test_wsospolyinterpmat_barrier(T::Type{<:HypReal})
    Random.seed!(1)
    for n in 1:3, halfdeg in 1:3, R in 1:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        P0 = convert(Matrix{T}, P0)
        cone = CO.WSOSPolyInterpMat{T}(R, U, [P0], true)
        test_barrier_oracles(cone)
    end
    return
end

function test_wsospolyinterpsoc_barrier(T::Type{<:HypReal})
    Random.seed!(1)
    for n in 1:2, halfdeg in 1:2, R in 3:3
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        P0 = convert(Matrix{T}, P0)
        cone = CO.WSOSPolyInterpSOC{T}(R, U, [P0], true)
        test_barrier_oracles(cone)
    end
    return
end
