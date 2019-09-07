#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Test
import Random
using LinearAlgebra
import ForwardDiff
import Hypatia
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities

function test_barrier_oracles(
    cone::CO.Cone{T},
    barrier::Function;
    noise::Real = 0,
    scale::Real = 1000,
    tol::Real = 100eps(T),
    ) where {T <: Real}
    CO.setup_data(cone)
    dim = CO.dimension(cone)
    point = Vector{T}(undef, dim)
    CO.set_initial_point(point, cone)
    if !iszero(noise)
        point += T(noise) * (rand(T, dim) .- inv(T(2)))
        point /= scale
    end
    CO.load_point(cone, point)
    @show point

    @test cone.point == point
    @test CO.is_feas(cone)
    nu = CO.get_nu(cone)
    grad = CO.grad(cone)
    @test dot(point, grad) ≈ -nu atol=tol rtol=tol
    hess = CO.hess(cone)
    @test hess * point ≈ -grad atol=tol rtol=tol

    if T in (Float32, Float64) # NOTE can only use BLAS floats with ForwardDiff barriers
        @test ForwardDiff.gradient(barrier, point) ≈ grad atol=tol rtol=tol
        @test ForwardDiff.hessian(barrier, point) ≈ hess atol=tol rtol=tol
    end

    inv_hess = CO.inv_hess(cone)
    @test hess * inv_hess ≈ I atol=tol rtol=tol

    CO.update_hess_prod(cone)
    CO.update_inv_hess_prod(cone)
    prod = similar(point)
    @test CO.hess_prod!(prod, point, cone) ≈ -grad atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod, grad, cone) ≈ -point atol=tol rtol=tol
    prod = similar(point, dim, dim)
    @test CO.hess_prod!(prod, inv_hess, cone) ≈ I atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod, hess, cone) ≈ I atol=tol rtol=tol
    id = Matrix{T}(I, dim, dim)
    @test CO.hess_prod!(prod, id, cone) ≈ hess atol=tol rtol=tol
    @test CO.inv_hess_prod!(prod, id, cone) ≈ inv_hess atol=tol rtol=tol

    return
end

function test_orthant_barrier(T::Type{<:Real})
    barrier = s -> -sum(log, s)
    for dim in [1, 3]
        cone = CO.Nonnegative{T}(dim)
        test_barrier_oracles(cone, barrier)
    end

    barrier = s -> -sum(log, -s)
    for dim in [1, 3]
        cone = CO.Nonpositive{T}(dim)
        test_barrier_oracles(cone, barrier)
    end
    return
end

function test_epinorminf_barrier(T::Type{<:Real})
    function barrier(s)
        u = s[1]
        w = s[2:end]
        return -sum(log(u - abs2(wj) / u) for wj in w) - log(u)
    end
    for dim in [2, 4]
        cone = CO.EpiNormInf{T}(dim)
        test_barrier_oracles(cone, barrier)
    end
    return
end

function test_epinormeucl_barrier(T::Type{<:Real})
    function barrier(s)
        u = s[1]
        w = s[2:end]
        return -log(abs2(u) - sum(abs2, w))
    end
    for dim in [2, 4]
        cone = CO.EpiNormEucl{T}(dim)
        test_barrier_oracles(cone, barrier)
    end
    return
end

function test_epipersquare_barrier(T::Type{<:Real})
    function barrier(s)
        u = s[1]
        v = s[2]
        w = s[3:end]
        return -log(2 * u * v - sum(abs2, w))
    end
    for dim in [3, 5]
        cone = CO.EpiPerSquare{T}(dim)
        test_barrier_oracles(cone, barrier)
    end
    return
end

function test_hypoperlog_barrier(T::Type{<:Real})
    function barrier(s)
        u = s[1]
        v = s[2]
        w = s[3:end]
        return -log(v * sum(log(wj / v) for wj in w) - u) - sum(log, w) - log(v)
    end
    for dim in [3, 5]
        cone = CO.HypoPerLog{T}(dim)
        test_barrier_oracles(cone, barrier)
    end
    return
end

function test_epiperexp_barrier(T::Type{<:Real})
    for dim in [3, 5]
        cone = CO.EpiPerExp{T}(dim)
        function barrier(s)
            u = s[1]
            v = s[2]
            w = s[3:end]
            return -log(v * log(u / v) - v * log(sum(wi -> exp(wi / v), w))) - log(u) - log(v)
        end
        test_barrier_oracles(cone, barrier)
    end
    return
end

function test_power_barrier(T::Type{<:Real})
    Random.seed!(1)
    for m in [2, 4], n in [1, 3]
        alpha = rand(T, m) .+ 1
        alpha ./= sum(alpha)
        cone = CO.Power{T}(alpha, n)
        function barrier(s)
            u = s[1:m]
            w = s[(m + 1):end]
            return -log(prod(u[j] ^ (2 * alpha[j]) for j in eachindex(u)) - sum(abs2, w)) - sum((1 - alpha[j]) * log(u[j]) for j in eachindex(u))
        end
        test_barrier_oracles(cone, barrier)
    end
    return
end

function test_hypogeomean_barrier(T::Type{<:Real})
    Random.seed!(1)
    for dim in [2, 4]
        alpha = rand(T, dim - 1) .+ 1
        alpha ./= sum(alpha)
        cone = CO.HypoGeomean{T}(alpha)
        function barrier(s)
            u = s[1]
            w = s[2:end]
            return -log(prod((w[j] / alpha[j]) ^ alpha[j] for j in eachindex(w)) + u) - sum((1 - alpha[j]) * log(w[j] / alpha[j]) for j in eachindex(w)) - log(-u)
        end
        test_barrier_oracles(cone, barrier)
    end
    return
end

function test_epinormspectral_barrier(T::Type{<:Real})
    for (n, m) in [(1, 2), (2, 2), (2, 3)]
        cone = CO.EpiNormSpectral{T}(n, m)
        function barrier(s)
            u = s[1]
            W = reshape(s[2:end], n, m)
            return -logdet(cholesky!(Symmetric(u * I - W * W' / u))) - log(u)
        end
        test_barrier_oracles(cone, barrier)
    end
    return
end

function test_possemideftri_barrier(T::Type{<:Real})
    for side in [1, 2, 3]
        # real PSD cone
        dim = div(side * (side + 1), 2)
        cone = CO.PosSemidefTri{T, T}(dim)
        function R_barrier(s)
            S = similar(s, side, side)
            CO.vec_to_mat_U!(S, s)
            return -logdet(cholesky!(Symmetric(S, :U)))
        end
        test_barrier_oracles(cone, R_barrier)

        # complex PSD cone
        dim = side^2
        cone = CO.PosSemidefTri{T, Complex{T}}(dim)
        function C_barrier(s)
            S = zeros(Complex{eltype(s)}, side, side)
            CO.vec_to_mat_U!(S, s)
            return -logdet(cholesky!(Hermitian(S, :U)))
        end
        test_barrier_oracles(cone, C_barrier)
    end
    return
end

function test_hypoperlogdettri_barrier(T::Type{<:Real})
    for side in [1, 2, 3]
        dim = 2 + div(side * (side + 1), 2)
        cone = CO.HypoPerLogdetTri{T}(dim)
        function barrier(s)
            u = s[1]
            v = s[2]
            W = similar(s, side, side)
            CO.vec_to_mat_U!(W, s[3:end])
            return -log(v * logdet(cholesky!(Symmetric(W / v, :U))) - u) - logdet(cholesky!(Symmetric(W, :U))) - log(v)
        end
        test_barrier_oracles(cone, barrier)
    end
    return
end

function test_wsospolyinterp_barrier(T::Type{<:Real})
    Random.seed!(1)
    for n in [1, 2, 3], halfdeg in [1, 2]
        (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
        P0 = convert(Matrix{T}, P0)
        cone = CO.WSOSPolyInterp{T, T}(U, [P0], true) # TODO test with more Pi
        function barrier(s)
            Lambda = Symmetric(P0' * Diagonal(s) * P0)
            return -logdet(cholesky!(Lambda))
        end
        test_barrier_oracles(cone, barrier)
    end
    # TODO also test complex case CO.WSOSPolyInterp{T, Complex{T}} - need complex MU interp functions first
    return
end

function test_wsospolymonomial_barrier(T::Type{<:Real})
    Random.seed!(1)
    for n in [2], halfdeg in [3]
        println()
        @show n, halfdeg
        cone = CO.WSOSPolyMonomial{T}(n, 2 * halfdeg)
        CO.setup_data(cone)
        barrier = cone.barfun
        test_barrier_oracles(cone, barrier, tol = 10_000 * eps(T))
    end
    return
end

function test_wsosconvexpolymonomial_barrier(T::Type{<:Real})
    Random.seed!(1)
    for n in [2, 3], deg in [4]
        println()
        @show n, deg
        cone = CO.WSOSConvexPolyMonomial{T}(n, deg)
        CO.setup_data(cone)
        barrier = cone.barfun
        test_barrier_oracles(cone, barrier, tol = 10_000 * eps(T))
    end
    return
end

# function test_wsospolyinterpmat_barrier(T::Type{<:Real})
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
# function test_wsospolyinterpsoc_barrier(T::Type{<:Real})
#     Random.seed!(1)
#     for n in 1:2, halfdeg in 1:2, R in 3:3
#         (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
#         P0 = convert(Matrix{T}, P0)
#         cone = CO.WSOSPolyInterpSOC{T}(R, U, [P0], true)
#         test_barrier_oracles(cone)
#     end
#     return
# end
