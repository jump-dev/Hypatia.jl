#=
tests for primitive cone barrier oracles
=#

using Test
import Random
import GenericLinearAlgebra.eigen
using LinearAlgebra
using SparseArrays
# import ForwardDiff # TODO not using barrier functions
import Hypatia
import Hypatia.ModelUtilities
import Hypatia.Cones

function test_barrier_oracles(
    cone::Cones.Cone{T},
    barrier::Function;
    noise::T = T(0.1),
    scale::T = T(1e-3),
    tol::Real = 1000eps(T),
    init_tol::Real = tol,
    init_only::Bool = false,
    ) where {T <: Real}
    Random.seed!(1)

    Cones.setup_data(cone)
    dim = Cones.dimension(cone)
    point = zeros(T, dim)
    dual_point = copy(point)
    Cones.set_initial_point(point, cone)
    Cones.set_initial_point(dual_point, cone)
    load_reset_check(cone, point, dual_point)
    @test cone.point == point
    @test cone.dual_point == dual_point

    if isfinite(init_tol)
        # tests for centrality of initial point
        minus_grad = -Cones.grad(cone)
        @test dot(point, minus_grad) ≈ norm(point) * norm(minus_grad) atol=init_tol rtol=init_tol
        @test point ≈ minus_grad atol=init_tol rtol=init_tol
        @test Cones.in_neighborhood(cone, one(T), one(T))
    end
    init_only && return

    # perturb and scale the initial point and check feasible
    perturb_scale(point, dual_point, noise, one(T))
    load_reset_check(cone, point, dual_point)

    # test gradient and Hessian oracles
    test_grad_hess(cone, point, dual_point, tol = tol)

    # check gradient and Hessian agree with ForwardDiff
    Cones.reset_data(cone)
    @test Cones.is_feas(cone)
    @test Cones.is_dual_feas(cone)
    grad = Cones.grad(cone)
    hess = Cones.hess(cone)
    # if dim < 8
    #     grad = Cones.grad(cone)
    #     fd_grad = ForwardDiff.gradient(barrier, point)
    #     @test grad ≈ fd_grad atol=tol rtol=tol
    #     hess = Cones.hess(cone)
    #     fd_hess = ForwardDiff.hessian(barrier, point)
    #     @test hess ≈ fd_hess atol=tol rtol=tol
    # end

    if Cones.use_correction(cone)
        # check correction satisfies log-homog property F'''(s)[s, s] = -2F''(s) * s = 2F'(s)
        @test -Cones.correction(cone, point) ≈ grad atol=tol rtol=tol
        # check correction term agrees with directional 3rd derivative
        (primal_dir, dual_dir) = perturb_scale(zeros(T, dim), zeros(T, dim), noise, one(T))
        corr = Cones.correction(cone, primal_dir)
        @test dot(corr, point) ≈ dot(primal_dir, hess * primal_dir) atol=tol rtol=tol

        # barrier_dir(point, t) = barrier(point + t * primal_dir)
        # @test -2 * corr ≈ ForwardDiff.gradient(x -> ForwardDiff.derivative(s -> ForwardDiff.derivative(t -> barrier_dir(x, t), s), 0), point) atol=tol rtol=tol
    end

    return
end

function test_grad_hess(cone::Cones.Cone{T}, point::Vector{T}, dual_point::Vector{T}; tol::Real = 1000eps(T)) where {T <: Real}
    # TODO not currently using dual_point
    nu = Cones.get_nu(cone)
    dim = length(point)
    grad = Cones.grad(cone)
    hess = Matrix(Cones.hess(cone))
    inv_hess = Matrix(Cones.inv_hess(cone))

    @test dot(point, grad) ≈ -nu atol=tol rtol=tol
    @test hess * inv_hess ≈ I atol=tol rtol=tol

    prod_mat = zeros(T, dim, dim)
    @test Cones.hess_prod!(prod_mat, inv_hess, cone) ≈ I atol=tol rtol=tol
    @test Cones.inv_hess_prod!(prod_mat, hess, cone) ≈ I atol=tol rtol=tol

    prod = zero(point)
    @test hess * point ≈ -grad atol=tol rtol=tol
    @test Cones.hess_prod!(prod, point, cone) ≈ -grad atol=tol rtol=tol
    @test Cones.inv_hess_prod!(prod, grad, cone) ≈ -point atol=tol rtol=tol

    if Cones.use_sqrt_oracles(cone)
        prod_mat2 = Matrix(Cones.hess_sqrt_prod!(prod_mat, inv_hess, cone)')
        @test Cones.hess_sqrt_prod!(prod_mat, prod_mat2, cone) ≈ I atol=tol rtol=tol
        Cones.inv_hess_sqrt_prod!(prod_mat2, Matrix(one(T) * I, dim, dim), cone)
        @test prod_mat2' * prod_mat2 ≈ inv_hess atol=tol rtol=tol
    end

    return
end

function load_reset_check(cone::Cones.Cone{T}, point::Vector{T}, dual_point::Vector{T}) where {T <: Real}
    Cones.load_point(cone, point)
    Cones.load_dual_point(cone, dual_point)
    Cones.reset_data(cone)
    @test Cones.is_feas(cone)
    @test Cones.is_dual_feas(cone)
    return
end

function perturb_scale(point::Vector{T}, dual_point::Vector{T}, noise::T, scale::T) where {T <: Real}
    if !iszero(noise)
        @. point += 2 * noise * rand(T) - noise
        @. dual_point += 2 * noise * rand(T) - noise
    end
    if !isone(scale)
        point .*= scale
    end
    return (point, dual_point)
end

# primitive cone barrier tests

function test_nonnegative_barrier(T::Type{<:Real})
    barrier = (s -> -sum(log, s))
    for dim in [1, 2, 3, 6]
        test_barrier_oracles(Cones.Nonnegative{T}(dim), barrier)
    end
    return
end

function test_epinorminf_barrier(T::Type{<:Real})
    for n in [1, 2, 3, 5]
        # real epinorminf cone
        function R_barrier(s)
            (u, w) = (s[1], s[2:end])
            return -sum(log(abs2(u) - abs2(wj)) for wj in w) + (n - 1) * log(u)
        end
        test_barrier_oracles(Cones.EpiNormInf{T, T}(1 + n), R_barrier)

        # complex epinorminf cone
        function C_barrier(s)
            (u, wr) = (s[1], s[2:end])
            w = Cones.rvec_to_cvec!(zeros(Complex{eltype(s)}, n), wr)
            return -sum(log(abs2(u) - abs2(wj)) for wj in w) + (n - 1) * log(u)
        end
        test_barrier_oracles(Cones.EpiNormInf{T, Complex{T}}(1 + 2n), C_barrier)
    end
    return
end

function test_epinormeucl_barrier(T::Type{<:Real})
    function barrier(s)
        (u, w) = (s[1], s[2:end])
        return -log(abs2(u) - sum(abs2, w))
    end
    for dim in [2, 3, 4, 6]
        test_barrier_oracles(Cones.EpiNormEucl{T}(dim), barrier)
    end
    return
end

function test_epipersquare_barrier(T::Type{<:Real})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(2 * u * v - sum(abs2, w))
    end
    for dim in [3, 4, 5, 7]
        test_barrier_oracles(Cones.EpiPerSquare{T}(dim), barrier)
    end
    return
end

function test_hypoperlog_barrier(T::Type{<:Real})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(v * sum(log(wj / v) for wj in w) - u) - sum(log, w) - log(v)
    end
    for dim in [3, 5, 10]
        test_barrier_oracles(Cones.HypoPerLog{T}(dim), barrier, init_tol = 1e-5)
    end
    for dim in [15, 65, 75, 100]
        test_barrier_oracles(Cones.HypoPerLog{T}(dim), barrier, init_tol = 1e-1, init_only = true)
    end
    return
end

function test_epiperentropy_barrier(T::Type{<:Real})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(u - sum(wi * log(wi / v) for wi in w)) - log(v) - sum(log(wi) for wi in w)
    end
    for w_dim in [1, 2, 3]
        test_barrier_oracles(Cones.EpiPerEntropy{T}(2 + w_dim), barrier, init_tol = 1e-5)
    end
    for w_dim in [15, 65, 75, 100]
        test_barrier_oracles(Cones.EpiPerEntropy{T}(2 + w_dim), barrier, init_tol = 1e-1, init_only = true)
    end
    return
end

function test_epipertraceentropytri_barrier(T::Type{<:Real})
    for side in [1, 2, 3, 12, 20]
        dim = 2 + Cones.svec_length(side)
        function barrier(s)
            (u, v, w) = (s[1], s[2], s[3:end])
            W = Hermitian(Cones.svec_to_smat!(zeros(eltype(s), side, side), w, sqrt(T(2))), :U)
            return -log(u - dot(W, log(W / v))) - log(v) - logdet(W)
        end
        if side <= 3
            test_barrier_oracles(Cones.EpiPerTraceEntropyTri{T}(dim), barrier, init_tol = 1e-5)
        else
            test_barrier_oracles(Cones.EpiPerTraceEntropyTri{T}(dim), barrier, init_tol = 1e-1, init_only = true)
        end
    end
    return
end

function test_epirelentropy_barrier(T::Type{<:Real})
    function barrier(s)
        (u, v, w) = (s[1], s[2:2:(end - 1)], s[3:2:end])
        return -log(u - sum(wi * log(wi / vi) for (vi, wi) in zip(v, w))) - sum(log(vi) + log(wi) for (vi, wi) in zip(v, w))
    end
    for w_dim in [1, 2, 3]
        test_barrier_oracles(Cones.EpiRelEntropy{T}(1 + 2 * w_dim), barrier, init_tol = 1e-5)
    end
    for w_dim in [15, 65, 75, 100]
        test_barrier_oracles(Cones.EpiRelEntropy{T}(1 + 2 * w_dim), barrier, init_tol = 1e-1, init_only = true)
    end
    return
end

function test_hypogeomean_barrier(T::Type{<:Real})
    for dim in [2, 3, 5, 8]
        invn = inv(T(dim - 1))
        function barrier(s)
            (u, w) = (s[1], s[2:end])
            return -log(exp(sum(invn * log(w[j]) for j in eachindex(w))) - u) - sum(log, w)
        end
        test_barrier_oracles(Cones.HypoGeoMean{T}(dim), barrier)
    end
    return
end

function test_hypopowermean_barrier(T::Type{<:Real})
    Random.seed!(1)
    for dim in [2, 3, 5, 15, 90, 120]
        alpha = rand(T, dim - 1) .+ 1
        alpha ./= sum(alpha)
        function barrier(s)
            (u, w) = (s[1], s[2:end])
            return -log(exp(sum(alpha[j] * log(w[j]) for j in eachindex(w))) - u) - sum(log, w)
        end
        cone = Cones.HypoPowerMean{T}(alpha)
        test_barrier_oracles(cone, barrier, init_tol = 1e-2, init_only = (dim > 5))
        # test initial point when all alphas are the same
        cone = Cones.HypoPowerMean{T}(fill(inv(T(dim - 1)), dim - 1))
        test_barrier_oracles(cone, barrier, init_tol = sqrt(eps(T)), init_only = true)
    end
    return
end

function test_power_barrier(T::Type{<:Real})
    Random.seed!(1)
    for (m, n) in [(2, 1), (2, 2), (4, 1), (2, 4)]
        alpha = rand(T, m) .+ 1
        alpha ./= sum(alpha)
        function barrier(s)
            (u, w) = (s[1:m], s[(m + 1):end])
            return -log(exp(2 * sum(alpha[j] * log(u[j]) for j in eachindex(u))) - sum(abs2, w)) - sum((1 - alpha[j]) * log(u[j]) for j in eachindex(u))
        end
        test_barrier_oracles(Cones.Power{T}(alpha, n), barrier)
    end
    return
end

function test_epinormspectral_barrier(T::Type{<:Real})
    for (n, m) in [(1, 1), (1, 2), (2, 2), (2, 4), (3, 4)]
        # real epinormspectral barrier
        function R_barrier(s)
            (u, W) = (s[1], reshape(s[2:end], n, m))
            return -logdet(cholesky!(Hermitian(abs2(u) * I - W * W'))) + (n - 1) * log(u)
        end
        test_barrier_oracles(Cones.EpiNormSpectral{T, T}(n, m), R_barrier)

        # complex epinormspectral barrier
        function C_barrier(s)
            u = s[1]
            W = Cones.rvec_to_cvec!(zeros(Complex{eltype(s)}, n, m), s[2:end])
            return -logdet(cholesky!(Hermitian(abs2(u) * I - W * W'))) + (n - 1) * log(u)
        end
        test_barrier_oracles(Cones.EpiNormSpectral{T, Complex{T}}(n, m), C_barrier)
    end
    return
end

function test_epitracerelentropytri_barrier(T::Type{<:Real})
    rt2 = sqrt(T(2))
    for side in [1, 2, 3, 8, 12]
        svec_dim = Cones.svec_length(side)
        cone = Cones.EpiTraceRelEntropyTri{T}(2 * svec_dim + 1)
        function barrier(s)
            u = s[1]
            u = s[1]
            V = Hermitian(Cones.svec_to_smat!(similar(s, side, side), s[2:(svec_dim + 1)], rt2), :U)
            W = Hermitian(Cones.svec_to_smat!(similar(s, side, side), s[(svec_dim + 2):end], rt2), :U)
            return -log(u - tr(W * log(W) - W * log(V))) - logdet(V) - logdet(W)
        end
        if side <= 3
            test_barrier_oracles(cone, barrier, init_tol = 1e-5)
        else
            test_barrier_oracles(cone, barrier, init_tol = 1e-1, init_only = true)
        end
    end
    return
end

function test_linmatrixineq_barrier(T::Type{<:Real})
    if !(T <: LinearAlgebra.BlasReal)
        return # TODO currently failing with BigFloat due to an apparent cholesky bug
    end
    Random.seed!(1)
    Rs_list = [[T, T], [Complex{T}, Complex{T}], [T, Complex{T}, T], [Complex{T}, T, T]]
    # Rs_list = [[T, T]]
    for side in [2, 3, 5], Rs in Rs_list
        As = Vector{LinearAlgebra.HermOrSym{R, Matrix{R}} where {R <: Hypatia.RealOrComplex{T}}}(undef, length(Rs))
        A_1_half = rand(Rs[1], side, side)
        As[1] = Hermitian(A_1_half * A_1_half' + I, :U)
        for i in 2:length(Rs)
            As[i] = Hermitian(rand(Rs[i], side, side), :U)
        end
        barrier(s) = -logdet(cholesky!(Hermitian(sum(s_i * A_i for (s_i, A_i) in zip(s, As)), :U)))
        test_barrier_oracles(Cones.LinMatrixIneq{T}(As), barrier, noise = T(1e-2), init_tol = Inf)
    end
    return
end

function test_possemideftri_barrier(T::Type{<:Real})
    for side in [1, 2, 3, 5]
        # real PSD cone
        function R_barrier(s)
            S = zeros(eltype(s), side, side)
            Cones.svec_to_smat!(S, s, sqrt(T(2)))
            return -logdet(cholesky!(Symmetric(S, :U)))
        end
        dim = Cones.svec_length(side)
        test_barrier_oracles(Cones.PosSemidefTri{T, T}(dim), R_barrier)

        # complex PSD cone
        function C_barrier(s)
            S = zeros(Complex{eltype(s)}, side, side)
            Cones.svec_to_smat!(S, s, sqrt(T(2)))
            return -logdet(cholesky!(Hermitian(S, :U)))
        end
        dim = side^2
        test_barrier_oracles(Cones.PosSemidefTri{T, Complex{T}}(dim), C_barrier)
    end
    return
end

function test_possemideftrisparse_barrier(T::Type{<:Real})
    if !(T <: LinearAlgebra.BlasReal)
        return # only works with BLAS real types
    end
    Random.seed!(1)
    invrt2 = inv(sqrt(T(2)))

    for side in [1, 2, 5, 10, 25, 50]
        # generate random sparsity pattern for lower triangle
        sparsity = inv(sqrt(side))
        (row_idxs, col_idxs, _) = findnz(tril!(sprand(Bool, side, side, sparsity)) + I)

        # real sparse PSD cone
        function R_barrier(s)
            scal_s = copy(s)
            for i in eachindex(s)
                if row_idxs[i] != col_idxs[i]
                    scal_s[i] *= invrt2
                end
            end
            S = Matrix(sparse(row_idxs, col_idxs, scal_s, side, side))
            return -logdet(cholesky(Symmetric(S, :L)))
        end
        test_barrier_oracles(Cones.PosSemidefTriSparse{T, T}(side, row_idxs, col_idxs), R_barrier)

        # complex sparse PSD cone
        function C_barrier(s)
            scal_s = zeros(Complex{eltype(s)}, length(row_idxs))
            idx = 1
            for i in eachindex(scal_s)
                if row_idxs[i] == col_idxs[i]
                    scal_s[i] = s[idx]
                    idx += 1
                else
                    scal_s[i] = invrt2 * Complex(s[idx], s[idx + 1])
                    idx += 2
                end
            end
            S = Matrix(sparse(row_idxs, col_idxs, scal_s, side, side))
            return -logdet(cholesky!(Hermitian(S, :L)))
        end
        test_barrier_oracles(Cones.PosSemidefTriSparse{T, Complex{T}}(side, row_idxs, col_idxs), C_barrier)
    end
    return
end

function test_doublynonnegativetri_barrier(T::Type{<:Real})
    for side in [1, 2, 3, 5]
        function barrier(s)
            S = zeros(eltype(s), side, side)
            Cones.svec_to_smat!(S, s, sqrt(T(2)))
            offdiags = vcat([div(i * (i - 1), 2) .+ (1:(i - 1)) for i in 2:side]...)
            return -logdet(cholesky!(Hermitian(S, :U))) - mapreduce(log, +, s[offdiags]; init = zero(eltype(s)))
        end
        dim = Cones.svec_length(side)
        test_barrier_oracles(Cones.DoublyNonnegativeTri{T}(dim), barrier, init_tol = sqrt(eps(T)), init_only = (side > 6))
    end
    return
end

function test_matrixepipersquare_barrier(T::Type{<:Real})
    for (n, m) in [(1, 1), (1, 2), (2, 2), (2, 4), (3, 4)]
        # real matrixepipersquare barrier
        per_idx = Cones.svec_length(n) + 1
        function R_barrier(s)
            U = Cones.svec_to_smat!(zeros(eltype(s), n, n), s[1:(per_idx - 1)], sqrt(T(2)))
            v = s[per_idx]
            W = reshape(s[(per_idx + 1):end], n, m)
            return -logdet(cholesky!(Symmetric(2 * v * U - W * W', :U))) + (n - 1) * log(v)
        end
        test_barrier_oracles(Cones.MatrixEpiPerSquare{T, T}(n, m), R_barrier)

        # complex matrixepipersquare barrier
        per_idx = n ^ 2 + 1
        function C_barrier(s)
            U = Cones.svec_to_smat!(zeros(Complex{eltype(s)}, n, n), s[1:(per_idx - 1)], sqrt(T(2)))
            v = s[per_idx]
            W = Cones.rvec_to_cvec!(zeros(Complex{eltype(s)}, n, m), s[(per_idx + 1):end])
            return -logdet(cholesky!(Hermitian(2 * v * U - W * W', :U))) + (n - 1) * log(v)
        end
        test_barrier_oracles(Cones.MatrixEpiPerSquare{T, Complex{T}}(n, m), C_barrier)
    end
    return
end

function test_hypoperlogdettri_barrier(T::Type{<:Real})
    for side in [1, 2, 3, 5, 8]
        # real logdet barrier
        dim = 2 + Cones.svec_length(side)
        cone = Cones.HypoPerLogdetTri{T, T}(dim)
        function R_barrier(s)
            (u, v, W) = (s[1], s[2], zeros(eltype(s), side, side))
            Cones.svec_to_smat!(W, s[3:end], sqrt(T(2)))
            return -log(v * logdet(cholesky!(Symmetric(W / v, :U))) - u) - logdet(cholesky!(Symmetric(W, :U))) - log(v)
        end
        if side <= 5
            test_barrier_oracles(cone, R_barrier, init_tol = 1e-5)
        else
            test_barrier_oracles(cone, R_barrier, init_tol = 1e-1, init_only = true)
        end

        # complex logdet barrier
        dim = 2 + side^2
        cone = Cones.HypoPerLogdetTri{T, Complex{T}}(dim)
        function C_barrier(s)
            (u, v, W) = (s[1], s[2], zeros(Complex{eltype(s)}, side, side))
            Cones.svec_to_smat!(W, s[3:end], sqrt(T(2)))
            return -log(v * logdet(cholesky!(Hermitian(W / v, :U))) - u) - logdet(cholesky!(Hermitian(W, :U))) - log(v)
        end
        if side <= 5
            test_barrier_oracles(cone, C_barrier, init_tol = 1e-5)
        else
            test_barrier_oracles(cone, C_barrier, init_tol = 1e-1, init_only = true)
        end
    end
    return
end

function test_hyporootdettri_barrier(T::Type{<:Real})
    for side in [1, 2, 3, 5, 8]
        # real rootdet barrier
        dim = 1 + Cones.svec_length(side)
        cone = Cones.HypoRootdetTri{T, T}(dim)
        function R_barrier(s)
            (u, W) = (s[1], zeros(eltype(s), side, side))
            Cones.svec_to_smat!(W, s[2:end], sqrt(T(2)))
            fact_W = cholesky!(Symmetric(W, :U))
            return -log(exp(logdet(fact_W) / side) - u) - logdet(fact_W)
        end
        test_barrier_oracles(cone, R_barrier)

        # complex rootdet barrier
        dim = 1 + side^2
        cone = Cones.HypoRootdetTri{T, Complex{T}}(dim)
        function C_barrier(s)
            (u, W) = (s[1], zeros(Complex{eltype(s)}, side, side))
            Cones.svec_to_smat!(W, s[2:end], sqrt(T(2)))
            fact_W = cholesky!(Hermitian(W, :U))
            return -log(exp(logdet(fact_W) / side) - u) - logdet(fact_W)
        end
        test_barrier_oracles(cone, C_barrier)
    end
    return
end

function test_wsosinterpnonnegative_barrier(T::Type{<:Real})
    Random.seed!(1)
    for (n, halfdeg) in [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1)]
        (U, _, Ps, _) = ModelUtilities.interpolate(ModelUtilities.Box{T}(-ones(T, n), ones(T, n)), halfdeg, sample = false) # use a unit box domain
        barrier(s) = -sum(logdet(cholesky!(Symmetric(P' * Diagonal(s) * P))) for P in Ps)
        cone = Cones.WSOSInterpNonnegative{T, T}(U, Ps)
        test_barrier_oracles(cone, barrier, init_tol = Inf) # TODO center and test initial points
    end
    # TODO also test complex case Cones.WSOSInterpNonnegative{T, Complex{T}} - need complex MU interp functions first
    return
end

function test_wsosinterppossemideftri_barrier(T::Type{<:Real})
    Random.seed!(1)
    rt2i = inv(sqrt(T(2)))
    for (n, halfdeg, R) in [(1, 1, 1), (1, 1, 4), (2, 2, 1), (2, 1, 3), (3, 1, 2)]
        (U, _, Ps, _) = ModelUtilities.interpolate(ModelUtilities.Box{T}(-ones(T, n), ones(T, n)), halfdeg, sample = false)
        cone = Cones.WSOSInterpPosSemidefTri{T}(R, U, Ps)
        function barrier(s)
            bar = zero(eltype(s))
            for P in Ps
                L = size(P, 2)
                Lambda = zeros(eltype(s), R * L, R * L)
                for i in 1:R, j in 1:i
                    Lambdaij = P' * Diagonal(s[Cones.block_idxs(U, Cones.svec_idx(i, j))]) * P
                    if i != j
                        Lambdaij .*= rt2i
                    end
                    Lambda[Cones.block_idxs(L, i), Cones.block_idxs(L, j)] = Lambdaij
                end
                bar -= logdet(cholesky!(Symmetric(Lambda, :L)))
            end
            return bar
        end
        test_barrier_oracles(cone, barrier, init_tol = Inf)
    end
    return
end

function test_wsosinterpepinormeucl_barrier(T::Type{<:Real})
    Random.seed!(1)
    for (n, halfdeg, R) in [(1, 1, 2), (1, 2, 4), (2, 2, 3), (3, 1, 2)]
        (U, _, Ps, _) = ModelUtilities.interpolate(ModelUtilities.Box{T}(-ones(T, n), ones(T, n)), halfdeg, sample = false)
        cone = Cones.WSOSInterpEpiNormEucl{T}(R, U, Ps)
        function barrier(s)
            bar = zero(eltype(s))
            for P in Ps
                Lambda = P' * Diagonal(s[1:U]) * P
                Lambda1fact = cholesky!(Symmetric(copy(Lambda), :L))
                for i in 2:R
                    Lambdai = P' * Diagonal(s[Cones.block_idxs(U, i)]) * P
                    ldiv!(Lambda1fact.L, Lambdai)
                    Lambda -= Lambdai' * Lambdai
                end
                bar -= logdet(cholesky!(Symmetric(Lambda))) + logdet(Lambda1fact)
            end
            return bar
        end
        test_barrier_oracles(cone, barrier, init_tol = Inf)
    end
    return
end

function test_wsosinterpepinormone_barrier(T::Type{<:Real})
    Random.seed!(1)
    for n in 1:3, halfdeg in 1:2, R in 2:3
        (U, _, Ps, _) = ModelUtilities.interpolate(ModelUtilities.Box{T}(-ones(T, n), ones(T, n)), halfdeg, sample = false)
        cone = Cones.WSOSInterpEpiNormOne{T}(R, U, Ps)
        function barrier(point)
            bar = zero(eltype(point))
            for P in cone.Ps
                lambda_1 = Symmetric(P' * Diagonal(point[1:U]) * P, :U)
                fact_1 = cholesky(lambda_1)
                for i in 2:R
                    lambda_i = Symmetric(P' * Diagonal(point[Cones.block_idxs(U, i)]) * P)
                    bar -= logdet(lambda_1 - lambda_i * (fact_1 \ lambda_i))
                end
                bar -= logdet(fact_1)
            end
            return bar
        end
        test_barrier_oracles(cone, barrier, init_tol = Inf)
    end
    return
end
