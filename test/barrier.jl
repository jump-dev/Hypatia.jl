#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using Test
import Random
using LinearAlgebra
import ForwardDiff
import Hypatia
const CO = Hypatia.Cones
# const MU = Hypatia.ModelUtilities

function test_barrier_oracles(cone::CO.Cone{T}, barrier::Function; noise = 0) where {T <: Real}
    CO.setup_data(cone)
    dim = CO.dimension(cone)
    point = Vector{T}(undef, dim)
    CO.set_initial_point(point, cone)
    if !iszero(noise)
        point += T(noise) * (rand(T, dim) .- inv(T(2)))
    end
    CO.load_point(cone, point)

    tol = 10000 * eps(T)

    @test cone.point == point
    @test CO.is_feas(cone)
    nu = CO.get_nu(cone)
    grad = CO.grad(cone)
    @test dot(point, grad) ≈ -nu atol=tol rtol=tol
    hess = CO.hess(cone)
    @test hess * point ≈ -grad atol=tol rtol=tol

    @show ForwardDiff.hessian(barrier, point) ./ hess
    if T in (Float32, Float64) # NOTE can only use BLAS floats with ForwardDiff barriers, see https://github.com/JuliaDiff/DiffResults.jl/pull/9#issuecomment-497853361
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
    for dim in [1, 3, 5]
        cone = CO.Nonnegative{T}(dim)
        test_barrier_oracles(cone, barrier)
        test_barrier_oracles(cone, barrier, noise = 0.1)
    end

    barrier = s -> -sum(log, -s)
    for dim in [1, 3, 5]
        cone = CO.Nonpositive{T}(dim)
        test_barrier_oracles(cone, barrier)
        test_barrier_oracles(cone, barrier, noise = 0.1)
    end
    return
end

function test_epinorminf_barrier(T::Type{<:Real})
    function barrier(s)
        u = s[1]
        w = s[2:end]
        return -sum(log, u .- abs2.(w) ./ u) - log(u)
    end
    for dim in [3, 5, 8]
        cone = CO.EpiNormInf{T}(dim)
        test_barrier_oracles(cone, barrier)
        test_barrier_oracles(cone, barrier, noise = 0.1)
    end
    return
end

function test_epinormeucl_barrier(T::Type{<:Real})
    function barrier(s)
        u = s[1]
        w = s[2:end]
        return -log(abs2(u) - sum(abs2, w))
    end
    for dim in [2, 3, 5]
        cone = CO.EpiNormEucl{T}(dim)
        test_barrier_oracles(cone, barrier)
        test_barrier_oracles(cone, barrier, noise = 0.1)
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
    for dim in [3, 5, 8]
        cone = CO.EpiPerSquare{T}(dim)
        test_barrier_oracles(cone, barrier)
        test_barrier_oracles(cone, barrier, noise = 0.1)
    end
    return
end

function test_epiperpower_barrier(T::Type{<:Real})
    for alpha in T[1.5, 2.5]
        cone = CO.EpiPerPower{T}(alpha)
        test_barrier_oracles(cone, cone.barfun)
        test_barrier_oracles(cone, cone.barfun, noise = 0.1)
    end
    return
end

function test_hypoperlog_barrier(T::Type{<:Real})
    function barrier(s)
        u = s[1]
        v = s[2]
        w = s[3:end]
        return -log(v * sum(log, w ./ v) - u) - sum(log, w) - log(v)
    end
    for dim in [3, 5, 8]
        cone = CO.HypoPerLog{T}(dim)
        test_barrier_oracles(cone, barrier)
        test_barrier_oracles(cone, barrier, noise = 0.1)
    end
    return
end

function test_epiperexp_barrier(T::Type{<:Real})
    for dim in [3, 5, 8]
        cone = CO.EpiPerExp{T}(dim)
        test_barrier_oracles(cone, cone.barfun)
        test_barrier_oracles(cone, cone.barfun, noise = 0.1)
    end
    return
end

function test_generalizedpower_barrier(T::Type{<:Real})
    Random.seed!(1)
    for dim in [3, 5, 8], m in [1, 2, 3]
        alpha = rand(T, dim - 1) .+ 1
        alpha ./= sum(alpha)
        cone = CO.GeneralizedPower{T}(alpha, 1)
        function barrier(s)
            u = s[1]
            w = s[2:end]
            return -log(prod(w.^(2 .* alpha)) - sum(abs2, u)) - sum((1 .- alpha) .* log.(w))
        end
        test_barrier_oracles(cone, barrier)
        test_barrier_oracles(cone, barrier, noise = 0.1)
    end
    return
end

function test_hypogeomean_barrier(T::Type{<:Real})
    Random.seed!(1)
    for dim in [3, 5, 8]
        alpha = rand(T, dim - 1) .+ 1
        alpha ./= sum(alpha)
        cone = CO.HypoGeomean{T}(alpha)
        function barrier(s)
            u = s[1]
            w = s[2:end]
            return -log(prod((w ./ alpha) .^ alpha) + u) - sum((1 .- alpha) .* log.(w ./ alpha)) - log(-u)
        end
        test_barrier_oracles(cone, barrier)
        test_barrier_oracles(cone, barrier, noise = 0.1)
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
        test_barrier_oracles(cone, barrier, noise = 0.1)
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
        test_barrier_oracles(cone, R_barrier, noise = 0.1)

        # complex PSD cone
        dim = side^2
        cone = CO.PosSemidefTri{T, Complex{T}}(dim)
        function C_barrier(s)
            S = zeros(Complex{eltype(s)}, side, side)
            CO.vec_to_mat_U!(S, s)
            return -logdet(cholesky!(Hermitian(S, :U)))
        end
        test_barrier_oracles(cone, C_barrier)
        test_barrier_oracles(cone, C_barrier, noise = 0.1)
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
        test_barrier_oracles(cone, barrier, noise = 0.1)
    end
    return
end

# function test_wsospolyinterp_barrier(T::Type{<:Real})
#     Random.seed!(1)
#     for n in 1:3, halfdeg in 1:3
#         (U, _, P0, _, _) = MU.interpolate(MU.FreeDomain(n), halfdeg, sample = false)
#         P0 = convert(Matrix{T}, P0)
#         cone = CO.WSOSPolyInterp{T, T}(U, [P0], true) # TODO test with more Pi
#         function barrier(s)
#             Lambda = Symmetric(P0' * Diagonal(s) * P0)
#             return -logdet(cholesky!(Lambda))
#         end
#         test_barrier_oracles(cone, barrier)
#         test_barrier_oracles(cone, barrier, noise = 0.1)
#     end
#     # TODO also test complex case CO.WSOSPolyInterp{T, Complex{T}} - need complex MU interp functions first
#     return
# end

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
