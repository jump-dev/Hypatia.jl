#=
tests for primitive cone barrier oracles
=#

using Test
import Random
import Random.randn
import GenericLinearAlgebra.eigen
using LinearAlgebra
using SparseArrays
import ForwardDiff
import Hypatia
import Hypatia.ModelUtilities
import Hypatia.Cones
import Hypatia.RealOrComplex

Random.randn(::Type{BigFloat}, dims::Integer...) = BigFloat.(randn(dims...))
Random.randn(::Type{Complex{BigFloat}}, dims::Integer...) = Complex{BigFloat}.(randn(ComplexF64, dims...))

# sanity check oracles
function test_oracles(
    cone::Cones.Cone{T};
    noise::T = T(1e-1),
    scale::T = T(1e-2),
    tol::Real = 1e3 * eps(T),
    init_only::Bool = false,
    init_tol::Real = tol,
    ) where {T <: Real}
    Random.seed!(1)
    dim = Cones.dimension(cone)
    Cones.setup_data(cone)
    Cones.reset_data(cone)

    point = zeros(T, dim)
    Cones.set_initial_point(point, cone)
    Cones.load_point(cone, point)
    @test Cones.is_feas(cone)
    @test cone.point == point

    dual_point = -Cones.grad(cone)
    Cones.load_dual_point(cone, dual_point)
    @test Cones.is_dual_feas(cone)
    @test cone.dual_point == dual_point
    # @test Cones.in_neighborhood(cone, one(T), one(T))

    # test centrality of initial point
    if isfinite(init_tol)
        @test point ≈ dual_point atol=init_tol rtol=init_tol
    end
    init_only && return

    # perturb and scale the initial point
    perturb_scale(point, noise, scale)
    perturb_scale(dual_point, noise, inv(scale))

    Cones.reset_data(cone)
    Cones.load_point(cone, point)
    @test Cones.is_feas(cone)
    Cones.load_dual_point(cone, dual_point)
    @test Cones.is_dual_feas(cone)

    # test gradient and Hessian oracles
    nu = Cones.get_nu(cone)
    grad = Cones.grad(cone)
    @test dot(point, grad) ≈ -nu atol=tol rtol=tol

    hess = Matrix(Cones.hess(cone))
    # inv_hess = Matrix(Cones.inv_hess(cone))
    # @test hess * inv_hess ≈ I atol=tol rtol=tol
    #
    prod_vec = zero(point)
    # @test hess * point ≈ -grad atol=tol rtol=tol
    @test Cones.hess_prod!(prod_vec, point, cone) ≈ -grad atol=tol rtol=tol
    # @test Cones.inv_hess_prod!(prod_vec, grad, cone) ≈ -point atol=tol rtol=tol
    #
    # prod_mat = zeros(T, dim, dim)
    # @test Cones.hess_prod!(prod_mat, inv_hess, cone) ≈ I atol=tol rtol=tol
    # @test Cones.inv_hess_prod!(prod_mat, hess, cone) ≈ I atol=tol rtol=tol
    #
    # if hasproperty(cone, :use_hess_prod_slow)
    #     Cones.update_use_hess_prod_slow(cone)
    #     @test cone.use_hess_prod_slow_updated
    #     @test !cone.use_hess_prod_slow
    #     cone.use_hess_prod_slow = true
    #     @test Cones.hess_prod_slow!(prod_mat, inv_hess, cone) ≈ I atol=tol rtol=tol
    # end
    #
    # if Cones.use_sqrt_hess_oracles(cone)
    #     prod_mat2 = Matrix(Cones.sqrt_hess_prod!(prod_mat, inv_hess, cone)')
    #     @test Cones.sqrt_hess_prod!(prod_mat, prod_mat2, cone) ≈ I atol=tol rtol=tol
    #     Cones.inv_sqrt_hess_prod!(prod_mat2, Matrix(one(T) * I, dim, dim), cone)
    #     @test prod_mat2' * prod_mat2 ≈ inv_hess atol=tol rtol=tol
    # end
    #
    # # test correction oracle
    # if Cones.use_correction(cone)
    #     @test -Cones.correction(cone, point) ≈ grad atol=tol rtol=tol
    #
    #     dir = perturb_scale(zeros(T, dim), noise, one(T))
    #     corr = Cones.correction(cone, dir)
    #     @test dot(corr, point) ≈ dot(dir, hess * dir) atol=tol rtol=tol
    # end

    return
end

# check some oracles agree with ForwardDiff
function test_barrier(
    cone::Cones.Cone{T},
    barrier::Function;
    noise::T = T(1e-1),
    scale::T = T(1e-2),
    tol::Real = 1e4 * eps(T),
    ) where {T <: Real}
    Random.seed!(1)
    dim = Cones.dimension(cone)
    Cones.setup_data(cone)

    point = zeros(T, dim)
    Cones.set_initial_point(point, cone)
    perturb_scale(point, noise, scale)

    Cones.reset_data(cone)
    Cones.load_point(cone, point)
    @test Cones.is_feas(cone)

    fd_grad = ForwardDiff.gradient(barrier, point)
    @test Cones.grad(cone) ≈ fd_grad atol=tol rtol=tol

    dir = 10 * randn(T, dim)
    barrier_dir(s, t) = barrier(s + t * dir)

    fd_hess_dir = ForwardDiff.gradient(s -> ForwardDiff.derivative(t -> barrier_dir(s, t), 0), point)
    Cones.hess(cone) # TODO remove
    # @test Cones.hess(cone) * dir ≈ fd_hess_dir atol=tol rtol=tol
    prod_vec = zero(dir)
    @test Cones.hess_prod!(prod_vec, dir, cone) ≈ fd_hess_dir atol=tol rtol=tol
    #
    # if Cones.use_correction(cone)
    #     fd_third_dir = ForwardDiff.gradient(s2 -> ForwardDiff.derivative(s -> ForwardDiff.derivative(t -> barrier_dir(s2, t), s), 0), point)
    #     @test -2 * Cones.correction(cone, dir) ≈ fd_third_dir atol=tol rtol=tol
    # end

    return
end

# show time and memory allocation for oracles
function show_time_alloc(
    cone::Cones.Cone{T};
    noise::T = T(1e-1),
    scale::T = T(1e-2),
    ) where {T <: Real}
    Random.seed!(1)
    dim = Cones.dimension(cone)
    println("dimension: ", dim)

    println("setup_data")
    @time Cones.setup_data(cone)
    Cones.reset_data(cone)

    point = zeros(T, dim)
    Cones.set_initial_point(point, cone)
    Cones.load_point(cone, point)
    @assert Cones.is_feas(cone)

    dual_point = -Cones.grad(cone)
    Cones.load_dual_point(cone, dual_point)
    @assert Cones.is_dual_feas(cone)

    Cones.reset_data(cone)

    Cones.load_point(cone, point)
    println("is_feas")
    @time Cones.is_feas(cone)

    Cones.load_dual_point(cone, dual_point)
    println("is_dual_feas")
    @time Cones.is_dual_feas(cone)

    println("grad")
    @time Cones.grad(cone)
    println("hess")
    @time Cones.hess(cone)
    println("inv_hess")
    @time Cones.inv_hess(cone)

    point1 = randn(T, dim)
    point2 = zero(point1)
    println("hess_prod")
    @time Cones.hess_prod!(point2, point1, cone)
    println("inv_hess_prod")
    @time Cones.inv_hess_prod!(point2, point1, cone)

    if hasproperty(cone, :use_hess_prod_slow)
        cone.use_hess_prod_slow_updated = true
        cone.use_hess_prod_slow = true
        println("hess_prod_slow")
        @time Cones.hess_prod_slow!(point2, point1, cone)
    end

    if Cones.use_sqrt_hess_oracles(cone)
        println("sqrt_hess_prod")
        @time Cones.sqrt_hess_prod!(point2, point1, cone)
        println("inv_sqrt_hess_prod")
        @time Cones.inv_sqrt_hess_prod!(point2, point1, cone)
    end

    if Cones.use_correction(cone)
        println("correction")
        @time Cones.correction(cone, point1)
    end

    println("in_neighborhood")
    @time Cones.in_neighborhood(cone, one(T), one(T))

    return
end

function perturb_scale(
    point::Vector{T},
    noise::T,
    scale::T,
    ) where {T <: Real}
    if !iszero(noise)
        @. point += 2 * noise * rand(T) - noise
    end
    if !isone(scale)
        point .*= scale
    end
    return point
end

# cone utilities

logdet_pd(W::Hermitian) = logdet(cholesky!(copy(W)))

# TODO maybe move to ModelUtilities

dim_vec(d::Int, ::Type{<:Real}) = d
dim_vec(d::Int, ::Type{<:Complex}) = 2 * d

dim_herm(d::Int, ::Type{<:Real}) = Cones.svec_length(d)
dim_herm(d::Int, ::Type{<:Complex}) = d^2

function new_vec(w::Vector, dw::Int, T::Type{<:Real})
    @assert length(w) == dw
    return w
end
function new_vec(w::Vector, dw::Int, R::Type{Complex{T}}) where {T <: Real}
    @assert length(w) == 2 * dw
    wR = zeros(Complex{eltype(w)}, dw)
    Cones.rvec_to_cvec!(wR, w)
    return wR
end

function new_mat_herm(w::Vector, dW::Int, T::Type{<:Real})
    @assert length(w) == dim_herm(dW, T)
    W = similar(w, dW, dW)
    Cones.svec_to_smat!(W, w, sqrt(T(2)))
    return Hermitian(W, :U)
end
function new_mat_herm(w::Vector, dW::Int, R::Type{Complex{T}}) where {T <: Real}
    @assert length(w) == dim_herm(dW, R)
    W = zeros(Complex{eltype(w)}, dW, dW)
    Cones.svec_to_smat!(W, w, sqrt(T(2)))
    return Hermitian(W, :U)
end

function rand_sppsd_pattern(dW::Int)
    sparsity = inv(sqrt(dW))
    (row_idxs, col_idxs, _) = findnz(tril!(sprand(Bool, dW, dW, sparsity)) + I)
    return (row_idxs, col_idxs)
end

function rand_herms(ds::Int, Rd::Vector, T::Type{<:Real})
    Ps = Vector{LinearAlgebra.HermOrSym{R, Matrix{R}} where {R <: RealOrComplex{T}}}(undef, length(Rd))
    A_1_half = randn(Rd[1], ds, ds)
    Ps[1] = Hermitian(A_1_half * A_1_half' + I, :U)
    for i in 2:length(Rd)
        Ps[i] = Hermitian(randn(Rd[i], ds, ds), :U)
    end
    return Ps
end

function rand_powers(T, d)
    Random.seed!(1)
    alpha = rand(T, d) .+ 1
    alpha ./= sum(alpha)
    return alpha
end

# real Ps for WSOS cones, use unit box domain
function rand_interp(num_vars::Int, halfdeg::Int, T::Type{<:Real})
    Random.seed!(1)
    domain = ModelUtilities.Box{T}(-ones(T, num_vars), ones(T, num_vars))
    (d, _, Ps, _) = ModelUtilities.interpolate(domain, halfdeg, sample = false)
    return (d, Ps)
end

# complex Ps for WSOS cones, use unit ball domain
function rand_interp(num_vars::Int, halfdeg::Int, R::Type{<:Complex{<:Real}})
    Random.seed!(1)
    gs = [z -> 1 - sum(abs2, z)]
    g_halfdegs = [1]
    (points, Ps) = ModelUtilities.interpolate(R, halfdeg, num_vars, gs, g_halfdegs)
    d = length(points)
    return (d, Ps)
end


# cones

# Nonnegative
function test_oracles(C::Type{<:Cones.Nonnegative})
    for d in [1, 2, 6]
        test_oracles(C(d))
    end
end

function test_barrier(C::Type{<:Cones.Nonnegative})
    barrier = (s -> -sum(log, s))
    test_barrier(C(3), barrier)
end

show_time_alloc(C::Type{<:Cones.Nonnegative}) = show_time_alloc(C(9))


# PosSemidefTri
function test_oracles(C::Type{Cones.PosSemidefTri{T, R}}) where {T, R}
    for dW in [1, 2, 3, 5]
        test_oracles(C(dim_herm(dW, R)))
    end
end

function test_barrier(C::Type{Cones.PosSemidefTri{T, R}}) where {T, R}
    dW = 3
    barrier(s) = -logdet_pd(Hermitian(new_mat_herm(s, dW, R), :U))
    test_barrier(C(dim_herm(dW, R)), barrier)
end

show_time_alloc(C::Type{Cones.PosSemidefTri{T, R}}) where {T, R} = show_time_alloc(C(dim_herm(4, R)))


# DoublyNonnegativeTri
function test_oracles(C::Type{Cones.DoublyNonnegativeTri{T}}) where T
    for dW in [1, 2, 5]
        test_oracles(C(Cones.svec_length(dW)), init_tol = sqrt(eps(T)))
    end
    for dW in [10, 20]
        test_oracles(C(Cones.svec_length(dW)), init_tol = sqrt(eps(T)), init_only = true)
    end
end

function test_barrier(C::Type{Cones.DoublyNonnegativeTri{T}}) where T
    dW = 3
    function barrier(s)
        W = new_mat_herm(s, dW, T)
        offdiags = vcat([div(i * (i - 1), 2) .+ (1:(i - 1)) for i in 2:dW]...)
        return -logdet_pd(Hermitian(W, :U)) - sum(log, s[offdiags])
    end
    test_barrier(C(Cones.svec_length(dW)), barrier)
end

show_time_alloc(C::Type{<:Cones.DoublyNonnegativeTri}) = show_time_alloc(C(10))


# PosSemidefTriSparse
function test_oracles(C::Type{<:Cones.PosSemidefTriSparse})
    for dW in [1, 2, 10, 25, 40]
        test_oracles(C(dW, rand_sppsd_pattern(dW)...))
    end
end

function test_barrier(C::Type{<:Cones.PosSemidefTriSparse{<:Cones.PSDSparseImpl, T, T}}) where T
    dW = 25
    (row_idxs, col_idxs) = rand_sppsd_pattern(dW)
    invrt2 = inv(sqrt(T(2)))
    function barrier(s)
        scal_s = copy(s)
        scal_s[row_idxs .!= col_idxs] .*= invrt2
        W = Matrix(sparse(row_idxs, col_idxs, scal_s, dW, dW))
        return -logdet_pd(Hermitian(W, :L))
    end
    test_barrier(C(dW, row_idxs, col_idxs), barrier)
end

function test_barrier(C::Type{<:Cones.PosSemidefTriSparse{<:Cones.PSDSparseImpl, T, Complex{T}}}) where T
    dW = 25
    (row_idxs, col_idxs) = rand_sppsd_pattern(dW)
    invrt2 = inv(sqrt(T(2)))
    function barrier(s)
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
        W = Matrix(sparse(row_idxs, col_idxs, scal_s, dW, dW))
        return -logdet_pd(Hermitian(W, :L))
    end
    test_barrier(C(dW, row_idxs, col_idxs), barrier)
end

show_time_alloc(C::Type{<:Cones.PosSemidefTriSparse}) = show_time_alloc(C(15, rand_sppsd_pattern(15)...))


# LinMatrixIneq
function test_oracles(C::Type{Cones.LinMatrixIneq{T}}) where T
    Random.seed!(1)
    Rd_list = [[T, T], [T, Complex{T}], [Complex{T}, T, T]]
    for ds in [2, 3, 4], Rd in Rd_list
        test_oracles(C(rand_herms(ds, Rd, T)), noise = T(1e-2), init_tol = Inf)
    end
end

function test_barrier(C::Type{Cones.LinMatrixIneq{T}}) where T
    Random.seed!(1)
    Ps = rand_herms(2, [T, Complex{T}], T)
    barrier(s) = -logdet_pd(Hermitian(sum(s[i] * Ps[i] for i in eachindex(s)), :U))
    test_barrier(C(Ps), barrier)
end

show_time_alloc(C::Type{Cones.LinMatrixIneq{T}}) where T = show_time_alloc(C(rand_herms(3, [T, Complex{T}, T, Complex{T}], T)))


# EpiNormInf
function test_oracles(C::Type{Cones.EpiNormInf{T, R}}) where {T, R}
    for dw in [1, 2, 5]
        test_oracles(C(1 + dim_vec(dw, R)))
    end
end

function test_barrier(C::Type{Cones.EpiNormInf{T, R}}) where {T, R}
    dw = 2
    function barrier(s)
        u = s[1]
        w = new_vec(s[2:end], dw, R)
        return -sum(log(abs2(u) - abs2(wi)) for wi in w) + (dw - 1) * log(u)
    end
    test_barrier(C(1 + dim_vec(dw, R)), barrier)
end

show_time_alloc(C::Type{<:Cones.EpiNormInf}) = show_time_alloc(C(9))


# EpiNormEucl
function test_oracles(C::Type{<:Cones.EpiNormEucl})
    for dw in [1, 2, 5]
        test_oracles(C(1 + dw))
    end
end

function test_barrier(C::Type{<:Cones.EpiNormEucl})
    function barrier(s)
        (u, w) = (s[1], s[2:end])
        return -log(abs2(u) - sum(abs2, w))
    end
    test_barrier(C(3), barrier)
end

show_time_alloc(C::Type{<:Cones.EpiNormEucl}) = show_time_alloc(C(9))


# EpiPerSquare
function test_oracles(C::Type{<:Cones.EpiPerSquare})
    for dw in [1, 2, 5]
        test_oracles(C(2 + dw))
    end
end

function test_barrier(C::Type{<:Cones.EpiPerSquare})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(2 * u * v - sum(abs2, w))
    end
    test_barrier(C(4), barrier)
end

show_time_alloc(C::Type{<:Cones.EpiPerSquare}) = show_time_alloc(C(9))


# EpiNormSpectral
function test_oracles(C::Type{<:Cones.EpiNormSpectral})
    for (dr, ds) in [(1, 1), (1, 2), (2, 2), (2, 4), (3, 4)]
        test_oracles(C(dr, ds))
    end
end

function test_barrier(C::Type{Cones.EpiNormSpectral{T, R}}) where {T, R}
    (dr, ds) = (2, 3)
    function barrier(s)
        u = s[1]
        W = reshape(new_vec(s[2:end], dr * ds, R), dr, ds)
        return -logdet_pd(Hermitian(abs2(u) * I - W * W')) + (dr - 1) * log(u)
    end
    test_barrier(C(dr, ds), barrier)
end

show_time_alloc(C::Type{<:Cones.EpiNormSpectral}) = show_time_alloc(C(2, 2))


# MatrixEpiPerSquare
function test_oracles(C::Type{<:Cones.MatrixEpiPerSquare})
    for (dr, ds) in [(1, 1), (1, 2), (2, 2), (2, 4), (3, 4)]
        test_oracles(C(dr, ds))
    end
end

function test_barrier(C::Type{Cones.MatrixEpiPerSquare{T, R}}) where {T, R}
    (dr, ds) = (2, 2)
    du = dim_herm(dr, R)
    function barrier(s)
        U = new_mat_herm(s[1:du], dr, R)
        v = s[du + 1]
        W = reshape(new_vec(s[(du + 2):end], dr * ds, R), dr, ds)
        return -logdet_pd(Hermitian(2 * v * U - W * W', :U)) + (dr - 1) * log(v)
    end
    test_barrier(C(dr, ds), barrier)
end

show_time_alloc(C::Type{<:Cones.MatrixEpiPerSquare}) = show_time_alloc(C(2, 2))


# GeneralizedPower
function test_oracles(C::Type{Cones.GeneralizedPower{T}}) where T
    for (du, dw) in [(2, 1), (3, 2), (4, 1), (2, 4)]
        test_oracles(C(rand_powers(T, du), dw))
    end
end

function test_barrier(C::Type{Cones.GeneralizedPower{T}}) where T
    (du, dw) = (2, 2)
    alpha = rand_powers(T, du)
    function barrier(s)
        (u, w) = (s[1:du], s[(du + 1):end])
        return -log(exp(2 * sum(alpha[i] * log(u[i]) for i in eachindex(u))) - sum(abs2, w)) - sum((1 - alpha[i]) * log(u[i]) for i in eachindex(u))
    end
    test_barrier(C(alpha, dw), barrier)
end

show_time_alloc(C::Type{Cones.GeneralizedPower{T}}) where T = show_time_alloc(C(rand_powers(T, 4), 5))


# HypoPowerMean
function test_oracles(C::Type{Cones.HypoPowerMean{T}}) where T
    for dw in [1, 2, 5]
        test_oracles(C(rand_powers(T, dw)), init_tol = 1e-2)
    end
    for dw in [15, 40, 100]
        test_oracles(C(rand_powers(T, dw)), init_tol = 1e-1, init_only = true)
    end
end

function test_barrier(C::Type{Cones.HypoPowerMean{T}}) where T
    alpha = rand_powers(T, 3)
    function barrier(s)
        (u, w) = (s[1], s[2:end])
        return -log(exp(sum(alpha[i] * log(w[i]) for i in eachindex(w))) - u) - sum(log, w)
    end
    test_barrier(C(alpha), barrier)
end

show_time_alloc(C::Type{Cones.HypoPowerMean{T}}) where T = show_time_alloc(C(rand_powers(T, 8)))


# HypoGeoMean
function test_oracles(C::Type{<:Cones.HypoGeoMean})
    for dw in [1, 2, 5]
        test_oracles(C(1 + dw))
    end
end

function test_barrier(C::Type{<:Cones.HypoGeoMean})
    function barrier(s)
        (u, w) = (s[1], s[2:end])
        sumlogw = sum(log, w)
        return -log(exp(sumlogw / length(w)) - u) - sumlogw
    end
    test_barrier(C(3), barrier)
end

show_time_alloc(C::Type{<:Cones.HypoGeoMean}) = show_time_alloc(C(9))


# HypoRootdetTri
function test_oracles(C::Type{Cones.HypoRootdetTri{T, R}}) where {T, R}
    for dW in [1, 2, 4]
        test_oracles(C(1 + dim_herm(dW, R)))
    end
end

function test_barrier(C::Type{Cones.HypoRootdetTri{T, R}}) where {T, R}
    dW = 3
    function barrier(s)
        (u, w) = (s[1], s[2:end])
        logdet_W = logdet_pd(new_mat_herm(w, dW, R))
        return -log(exp(logdet_W / dW) - u) - logdet_W
    end
    test_barrier(C(1 + dim_herm(dW, R)), barrier)
end

show_time_alloc(C::Type{Cones.HypoRootdetTri{T, R}}) where {T, R} = show_time_alloc(C(1 + dim_herm(3, R)))







# EpiPerSepSpectral
function test_oracles(C::Type{<:Cones.EpiPerSepSpectral})
    for d in [1, 2, 3, 6]
        test_oracles(C(d), init_tol = Inf)
    end
end

# TODO other CSqr
function test_barrier(C::Type{<:Cones.EpiPerSepSpectral{<:Cones.VectorCSqr, F}}) where F
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(u - v * Cones.h_val(F, w ./ v)) - log(v) - sum(log, w)
    end
    test_barrier(C(3), barrier)
end

function test_barrier(C::Type{<:Cones.EpiPerSepSpectral{<:Cones.MatrixCSqr{T, R}, F}}) where {T, R, F}
    dW = 3
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        W = new_mat_herm(w, dW, R)
        Wλ = eigen(W).values
        return -log(u - v * Cones.h_val(F, Wλ ./ v)) - log(v) - sum(log, Wλ)
    end
    test_barrier(C(dW), barrier)
end

# TODO pick sizes
show_time_alloc(C::Type{<:Cones.EpiPerSepSpectral{<:Cones.VectorCSqr}}) = show_time_alloc(C(1000))
show_time_alloc(C::Type{<:Cones.EpiPerSepSpectral{<:Cones.MatrixCSqr}}) = show_time_alloc(C(30))








# HypoPerLog
function test_oracles(C::Type{<:Cones.HypoPerLog})
    for dw in [1, 2, 5]
        test_oracles(C(2 + dw), init_tol = 1e-5)
    end
    for dw in [15, 40, 100]
        test_oracles(C(2 + dw), init_tol = 1e-1, init_only = true)
    end
end

function test_barrier(C::Type{<:Cones.HypoPerLog})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(v * sum(log(wi / v) for wi in w) - u) - log(v) - sum(log, w)
    end
    test_barrier(C(4), barrier)
end

show_time_alloc(C::Type{<:Cones.HypoPerLog}) = show_time_alloc(C(9))


# HypoPerLogdetTri
function test_oracles(C::Type{Cones.HypoPerLogdetTri{T, R}}) where {T, R}
    for dW in [1, 2, 4]
        test_oracles(C(2 + dim_herm(dW, R)), init_tol = 1e-4)
    end
    for dW in [8, 12]
        test_oracles(C(2 + dim_herm(dW, R)), init_tol = 1e-1, init_only = true)
    end
end

function test_barrier(C::Type{Cones.HypoPerLogdetTri{T, R}}) where {T, R}
    dW = 3
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        W = new_mat_herm(w, dW, R)
        return -log(v * logdet_pd(W / v) - u) - log(v) - logdet_pd(W)
    end
    test_barrier(C(2 + dim_herm(dW, R)), barrier)
end

show_time_alloc(C::Type{Cones.HypoPerLogdetTri{T, R}}) where {T, R} = show_time_alloc(C(2 + dim_herm(3, R)))


# EpiPerEntropy
function test_oracles(C::Type{<:Cones.EpiPerEntropy})
    for dw in [1, 2, 5]
        test_oracles(C(2 + dw), init_tol = 1e-5)
    end
    for dw in [15, 40, 100]
        test_oracles(C(2 + dw), init_tol = 1e-1, init_only = true)
    end
end

function test_barrier(C::Type{<:Cones.EpiPerEntropy})
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        return -log(u - sum(wi * log(wi / v) for wi in w)) - log(v) - sum(log, w)
    end
    test_barrier(C(4), barrier)
end

show_time_alloc(C::Type{<:Cones.EpiPerEntropy}) = show_time_alloc(C(9))


# EpiPerTrEntropyTri
function test_oracles(C::Type{<:Cones.EpiPerTrEntropyTri})
    for dW in [1, 2, 4]
        test_oracles(C(2 + Cones.svec_length(dW)), init_tol = 1e-4)
    end
    for dW in [8, 12]
        test_oracles(C(2 + Cones.svec_length(dW)), init_tol = 1e-1, init_only = true)
    end
end

function test_barrier(C::Type{Cones.EpiPerTrEntropyTri{T}}) where T
    dW = 3
    function barrier(s)
        (u, v, w) = (s[1], s[2], s[3:end])
        W = new_mat_herm(w, dW, T)
        return -log(u - dot(W, log(W / v))) - log(v) - logdet_pd(W)
    end
    test_barrier(C(2 + Cones.svec_length(dW)), barrier)
end

show_time_alloc(C::Type{<:Cones.EpiPerTrEntropyTri}) = show_time_alloc(C(8))


# EpiRelEntropy
function test_oracles(C::Type{<:Cones.EpiRelEntropy})
    for dw in [1, 2, 4]
        test_oracles(C(1 + 2 * dw), init_tol = 1e-5)
    end
    for dw in [15, 40, 100]
        test_oracles(C(1 + 2 * dw), init_tol = 1e-1, init_only = true)
    end
end

function test_barrier(C::Type{<:Cones.EpiRelEntropy})
    dw = 2
    function barrier(s)
        (u, v, w) = (s[1], s[2:(1 + dw)], s[(2 + dw):end])
        return -log(u - sum(wi * log(wi / vi) for (vi, wi) in zip(v, w))) - sum(log, v) - sum(log, w)
    end
    test_barrier(C(5), barrier)
end

show_time_alloc(C::Type{<:Cones.EpiRelEntropy}) = show_time_alloc(C(9))


# EpiTrRelEntropyTri
function test_oracles(C::Type{<:Cones.EpiTrRelEntropyTri})
    for dW in [1, 2, 4]
        test_oracles(C(1 + 2 * Cones.svec_length(dW)), init_tol = 1e-4)
    end
    for dW in [6, 10]
        test_oracles(C(1 + 2 * Cones.svec_length(dW)), init_tol = 1e-1, init_only = true)
    end
end

function test_barrier(C::Type{Cones.EpiTrRelEntropyTri{T}}) where T
    dW = 3
    dw = Cones.svec_length(dW)
    function barrier(s)
        (u, v, w) = (s[1], s[1 .+ (1:dw)], s[(2 + dw):end])
        V = new_mat_herm(v, dW, T)
        W = new_mat_herm(w, dW, T)
        return -log(u - dot(W, log(W) - log(V))) - logdet_pd(V) - logdet_pd(W)
    end
    test_barrier(C(1 + 2 * dw), barrier, tol = 1e8 * eps(T))
end

show_time_alloc(C::Type{<:Cones.EpiTrRelEntropyTri}) = show_time_alloc(C(13))


# WSOSInterpNonnegative
function test_oracles(C::Type{Cones.WSOSInterpNonnegative{T, R}}) where {T, R}
    for (num_vars, halfdeg) in [(1, 1), (1, 3), (2, 1), (2, 2), (3, 1)]
        (d, Ps) = rand_interp(num_vars, halfdeg, R)
        test_oracles(C(d, Ps), init_tol = Inf)
    end
end

function test_barrier(C::Type{Cones.WSOSInterpNonnegative{T, R}}) where {T, R}
    (d, Ps) = rand_interp(2, 1, R)
    barrier(s) = -sum(logdet_pd(Hermitian(P' * Diagonal(s) * P)) for P in Ps)
    test_barrier(C(d, Ps), barrier)
end

show_time_alloc(C::Type{Cones.WSOSInterpNonnegative{T, R}}) where {T, R} = show_time_alloc(C(rand_interp(3, 1, R)...))


# WSOSInterpPosSemidefTri
function test_oracles(C::Type{Cones.WSOSInterpPosSemidefTri{T}}) where T
    for (num_vars, halfdeg, ds) in [(1, 1, 1), (1, 1, 4), (2, 2, 1), (2, 1, 3), (3, 1, 2)]
        (d, Ps) = rand_interp(num_vars, halfdeg, T)
        test_oracles(C(ds, d, Ps), init_tol = Inf)
    end
end

function test_barrier(C::Type{Cones.WSOSInterpPosSemidefTri{T}}) where T
    (d, Ps) = rand_interp(1, 1, T)
    ds = 3
    invrt2 = inv(sqrt(T(2)))
    function barrier(s)
        function ldlamP(P)
            dt = size(P, 2)
            lam = similar(s, ds * dt, ds * dt)
            for i in 1:ds, j in 1:i
                lamij = P' * Diagonal(s[Cones.block_idxs(d, Cones.svec_idx(i, j))]) * P
                if i != j
                    lamij .*= invrt2
                end
                lam[Cones.block_idxs(dt, i), Cones.block_idxs(dt, j)] = lamij
            end
            return -logdet_pd(Hermitian(lam, :L))
        end
        return sum(ldlamP, Ps)
    end
    test_barrier(C(ds, d, Ps), barrier)
end

show_time_alloc(C::Type{Cones.WSOSInterpPosSemidefTri{T}}) where T = show_time_alloc(C(2, rand_interp(2, 1, T)...))


# WSOSInterpEpiNormEucl
function test_oracles(C::Type{Cones.WSOSInterpEpiNormEucl{T}}) where T
    for (num_vars, halfdeg, ds) in [(1, 1, 1), (1, 2, 3), (2, 2, 2), (3, 1, 1)]
        (d, Ps) = rand_interp(num_vars, halfdeg, T)
        test_oracles(C(1 + ds, d, Ps), init_tol = Inf)
    end
end

function test_barrier(C::Type{Cones.WSOSInterpEpiNormEucl{T}}) where T
    (d, Ps) = rand_interp(1, 1, T)
    ds = 2
    invrt2 = inv(sqrt(T(2)))
    function barrier(s)
        function ldlamP(P)
            lam = P' * Diagonal(s[1:d]) * P
            lam1fact = cholesky(Hermitian(lam, :L))
            PL1 = lam1fact.L \ P'
            for i in 1:ds
                lamLi = PL1 * Diagonal(s[Cones.block_idxs(d, 1 + i)]) * P
                lam -= lamLi' * lamLi
            end
            return -logdet(lam1fact) - logdet_pd(Hermitian(lam))
        end
        return sum(ldlamP, Ps)
    end
    test_barrier(C(1 + ds, d, Ps), barrier)
end

show_time_alloc(C::Type{Cones.WSOSInterpEpiNormEucl{T}}) where T = show_time_alloc(C(3, rand_interp(2, 1, T)...))


# WSOSInterpEpiNormOne
function test_oracles(C::Type{Cones.WSOSInterpEpiNormOne{T}}) where T
    for (num_vars, halfdeg, ds) in [(1, 1, 1), (1, 2, 3), (2, 2, 2), (3, 1, 1)]
        (d, Ps) = rand_interp(num_vars, halfdeg, T)
        test_oracles(C(1 + ds, d, Ps), init_tol = Inf)
    end
end

function test_barrier(C::Type{Cones.WSOSInterpEpiNormOne{T}}) where T
    (d, Ps) = rand_interp(1, 1, T)
    ds = 2
    invrt2 = inv(sqrt(T(2)))
    function barrier(s)
        function ldlamP(P)
            lam = P' * Diagonal(s[1:d]) * P
            lam1fact = cholesky(Hermitian(lam, :L))
            PL1 = lam1fact.L \ P'
            lamLs = [PL1 * Diagonal(s[Cones.block_idxs(d, 1 + i)]) * P for i in 1:ds]
            lams = [Hermitian(lam - lamLi' * lamLi) for lamLi in lamLs]
            return -logdet(lam1fact) - sum(logdet_pd, lams)
        end
        return sum(ldlamP, Ps)
    end
    test_barrier(C(1 + ds, d, Ps), barrier)
end

show_time_alloc(C::Type{Cones.WSOSInterpEpiNormOne{T}}) where T = show_time_alloc(C(3, rand_interp(2, 1, T)...))
