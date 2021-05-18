"""
$(TYPEDEF)

Hypograph of real symmetric or complex Hermitian root-determinant cone of
dimension `dim` in svec format.

    $(FUNCTIONNAME){T, R}(dim::Int, use_dual::Bool = false)
"""
mutable struct HypoRootdetTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
    is_complex::Bool
    rt2::T

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    mat::Matrix{R}
    mat2::Matrix{R}
    mat3::Matrix{R}
    fact_W
    rtdet::T
    z::T
    sigma::T
    dot_const::T
    W::Matrix{R}
    Wi::Matrix{R}
    Wi_vec::Vector{T}
    tempw::Vector{T}

    function HypoRootdetTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {R <: RealOrComplex{T}} where {T <: Real}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        if R <: Complex
            d = isqrt(dim - 1) # real lower triangle and imaginary under diagonal
            @assert d^2 == dim - 1
            cone.is_complex = true
        else
            d = round(Int, sqrt(0.25 + 2 * (dim - 1)) - 0.5)
            @assert d * (d + 1) == 2 * (dim - 1)
            cone.is_complex = false
        end
        cone.d = d
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data!(
    cone::HypoRootdetTri{T, R},
    ) where {R <: RealOrComplex{T}} where {T <: Real}
    dim = cone.dim


    # don't alloc here
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)



    d = cone.d
    cone.mat = zeros(R, d, d)
    cone.mat2 = zeros(R, d, d)
    cone.mat3 = zeros(R, d, d)
    cone.W = zeros(R, d, d)
    cone.Wi_vec = zeros(T, dim - 1)
    cone.tempw = zeros(T, dim - 1)
    return cone
end

get_nu(cone::HypoRootdetTri) = (cone.d + 1)

function set_initial_point!(
    arr::AbstractVector{T},
    cone::HypoRootdetTri{T, R},
    ) where {R <: RealOrComplex{T}} where {T <: Real}
    arr .= 0
    d = cone.d
    const1 = sqrt(T(5d^2 + 2d + 1))
    const2 = arr[1] = -sqrt((3d + 1 - const1) / T(2d + 2))
    const3 = -const2 * (d + 1 + const1) / 2d
    incr = (cone.is_complex ? 2 : 1)
    k = 2
    @inbounds for i in 1:d
        arr[k] = const3
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::HypoRootdetTri{T}) where T
    @assert !cone.feas_updated

    @views svec_to_smat!(cone.mat, cone.point[2:end], cone.rt2)
    fact = cone.fact_W = cholesky!(Hermitian(cone.mat, :U), check = false)
    if isposdef(fact)
        cone.rtdet = exp(logdet(fact) / cone.d)
        cone.z = cone.rtdet - cone.point[1]
        cone.is_feas = (cone.z > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoRootdetTri{T}) where T
    u = cone.dual_point[1]
    if u < -eps(T)
        @views svec_to_smat!(cone.mat2, cone.dual_point[2:end], cone.rt2)
        fact = cholesky!(Hermitian(cone.mat2, :U), check = false)
        return isposdef(fact) && (logdet(fact) -
            cone.d * log(-u / cone.d) > eps(T))
    end
    return false
end

function update_grad(
    cone::HypoRootdetTri{T, R},
    ) where {R <: RealOrComplex{T}} where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    g = cone.grad

    g[1] = inv(cone.z)
    # TODO in-place
    # copyto!(cone.Wi, cone.fact_W.factors)
    # LinearAlgebra.inv!(Cholesky(cone.Wi, 'U', 0)) # TODO inplace for bigfloat
    cone.Wi = inv(cone.fact_W)
    smat_to_svec!(cone.Wi_vec, cone.Wi, cone.rt2)
    sigma = cone.sigma = cone.rtdet / (cone.z * cone.d)
    g2const = -(sigma + 1)
    @. @views g[2:end] = g2const * cone.Wi_vec
    cone.dot_const = sigma * (sigma - inv(T(cone.d)))

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoRootdetTri)
    @assert cone.grad_updated
    H = cone.hess.data
    Wi_vec = cone.Wi_vec
    z = cone.z
    sigma = cone.sigma
    Huwconst = -sigma / z
    sckron = sigma + 1
    dot_const = cone.dot_const


    # @warn("don't store it in hessian")

    @inbounds begin
        H[1, 1] = inv(z) / z
        @. @views H[1, 2:end] = Huwconst * Wi_vec
    end

    @inbounds @views symm_kron!(H[2:end, 2:end], cone.Wi, cone.rt2)
    @inbounds for j in eachindex(Wi_vec)
        j1 = 1 + j
        Wi_vecj = dot_const * Wi_vec[j]
        for i in 1:j
            H[1 + i, j1] = sckron * H[1 + i, j1] + Wi_vec[i] * Wi_vecj
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoRootdetTri,
    )
    @assert cone.grad_updated
    Wi_vec = cone.Wi_vec
    z = cone.z
    sigma = cone.sigma
    sigmap1 = sigma + 1
    const_diag = cone.dot_const / sigmap1
    Huwconst = -sigma / z

    @inbounds for i in 1:size(arr, 2)
        arr_u = arr[1, i]
        @views arr_w = arr[2:end, i]
        @views prod_w = prod[2:end, i]
        prod[1, i] = (arr_u / z - sigma * dot(Wi_vec, arr_w)) / z
        svec_to_smat!(cone.mat2, arr_w, cone.rt2)
        copytri!(cone.mat2, 'U', true)
        rdiv!(cone.mat2, cone.fact_W)
        const_i = tr(cone.mat2) * const_diag
        for j in 1:cone.d
            cone.mat2[j, j] += const_i
        end
        ldiv!(cone.fact_W, cone.mat2)
        smat_to_svec!(prod_w, cone.mat2, cone.rt2)
        @. prod_w *= sigmap1
        const_i = arr_u * Huwconst
        @. prod_w += const_i * Wi_vec
    end

    return prod
end

function update_inv_hess(cone::HypoRootdetTri)
    @views w = cone.point[2:end]
    svec_to_smat!(cone.W, w, cone.rt2)
    W = Hermitian(cone.W, :U)
    Hi = cone.inv_hess.data
    z = cone.z
    d = cone.d
    rtdet = cone.rtdet
    den = d * z + rtdet
    scdot = rtdet / (d * den)
    sckron = z * d / den

    Hi[1, 1] = abs2(z) + abs2(rtdet) / d
    Hi12const = rtdet / d
    @. @views Hi[1, 2:end] = Hi12const * w

    @inbounds @views symm_kron!(Hi[2:end, 2:end], W, cone.rt2)
    @inbounds for j in eachindex(w)
        j1 = 1 + j
        scwj = scdot * w[j]
        for i in 1:j
            Hi[1 + i, j1] = sckron * Hi[1 + i, j1] + w[i] * scwj
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoRootdetTri,
    )
    @views w = cone.point[2:end]
    svec_to_smat!(cone.W, w, cone.rt2)
    W = Hermitian(cone.W, :U)
    z = cone.z
    d = cone.d
    rtdet = cone.rtdet
    const0 = d * z + rtdet
    const1 = rtdet / d
    const2 = const1 / const0
    const3 = d * z / const0
    const4 = abs2(z) + const1 * rtdet

    @inbounds for i in 1:size(arr, 2)
        arr_u = arr[1, i]
        @views arr_w = arr[2:end, i]
        @views prod_w = prod[2:end, i]
        svec_to_smat!(cone.mat2, arr_w, cone.rt2)
        copytri!(cone.mat2, 'U', true)
        mul!(cone.mat3, cone.mat2, W)
        mul!(cone.mat2, W, cone.mat3)
        smat_to_svec!(prod_w, cone.mat2, cone.rt2)
        @. prod_w *= const3
        dot_i = dot(w, arr_w)
        const_i = const2 * dot_i + const1 * arr_u
        @. prod_w += const_i * w
        prod[1, i] = const1 * dot_i + const4 * arr_u
    end

    return prod
end

function dder3(cone::HypoRootdetTri{T}, dir::AbstractVector{T}) where T
    @assert cone.grad_updated
    u_dir = dir[1]
    @views w_dir = dir[2:end]
    dder3 = cone.dder3
    @views w_dder3 = dder3[2:end]
    sigma = cone.sigma
    z = cone.z
    Wi_vec = cone.Wi_vec

    S = copytri!(svec_to_smat!(cone.mat3, w_dir, cone.rt2), 'U', true)
    dot_Wi_S = dot(Wi_vec, w_dir)
    ldiv!(cone.fact_W, S)
    dot_skron = real(dot(S, S'))

    rdiv!(S, cone.fact_W.U)
    mul!(cone.mat2, S, S')
    @views smat_to_svec!(w_dder3, cone.mat2, cone.rt2)
    w_dder3 .*= -2 * (sigma + 1)

    ssigma = inv(T(cone.d)) - sigma
    scal1 = sigma * ssigma
    scal2 = dot_Wi_S * (ssigma - sigma)
    udz = u_dir / z
    scal4 = 2 * udz * sigma
    scal5 = scal1 * (dot_skron - scal2 * dot_Wi_S) - scal4 * (scal2 + udz)
    scal6 = 2 * dot_Wi_S * scal1 + scal4
    @. w_dder3 += scal5 * Wi_vec

    skron2 = rdiv!(S, cone.fact_W.U')
    vec_skron2 = smat_to_svec!(cone.tempw, skron2, cone.rt2)

    @. w_dder3 += scal6 * vec_skron2
    dder3[1] = (sigma * (dot(vec_skron2, w_dir) - (scal2 + 4 * udz) * dot_Wi_S) +
        2 * abs2(udz)) / z
    dder3 ./= -2

    return dder3
end
