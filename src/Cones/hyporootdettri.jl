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
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    di::T
    ϕ::T
    ζ::T
    ϕζidi::T
    mat::Matrix{R}
    fact_W::Cholesky{R}
    Wi::Matrix{R}
    Wi_vec::Vector{T}
    tempw::Vector{T}
    mat2::Matrix{R}
    mat3::Matrix{R}
    mat4::Matrix{R}

    function HypoRootdetTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert dim >= 2
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        cone.is_complex = (R <: Complex)
        cone.d = svec_side(R, dim - 1)
        return cone
    end
end

reset_data(cone::HypoRootdetTri) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

function setup_extra_data!(
    cone::HypoRootdetTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    dim = cone.dim
    d = cone.d
    cone.di = inv(T(d))
    cone.mat = zeros(R, d, d)
    cone.Wi = zeros(R, d, d)
    cone.Wi_vec = zeros(T, dim - 1)
    cone.tempw = zeros(T, dim - 1)
    cone.mat2 = zeros(R, d, d)
    cone.mat3 = zeros(R, d, d)
    cone.mat4 = zeros(R, d, d)
    return cone
end

get_nu(cone::HypoRootdetTri) = 1 + cone.d

function set_initial_point!(
    arr::AbstractVector{T},
    cone::HypoRootdetTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    arr .= 0
    d = cone.d
    c1 = sqrt(T(5d^2 + 2d + 1))
    c2 = arr[1] = -sqrt((3d + 1 - c1) / T(2d + 2))
    c3 = -c2 * (d + 1 + c1) / 2d
    incr = (cone.is_complex ? 2 : 1)
    k = 2
    @inbounds for i in 1:d
        arr[k] = c3
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::HypoRootdetTri{T}) where {T <: Real}
    @assert !cone.feas_updated

    @views svec_to_smat!(cone.mat, cone.point[2:end], cone.rt2)
    fact = cone.fact_W = cholesky!(Hermitian(cone.mat, :U), check = false)
    if isposdef(fact)
        cone.ϕ = exp(logdet(fact) / cone.d)
        cone.ζ = cone.ϕ - cone.point[1]
        cone.is_feas = (cone.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoRootdetTri{T}) where {T <: Real}
    u = cone.dual_point[1]

    if u < -eps(T)
        @views svec_to_smat!(cone.mat2, cone.dual_point[2:end], cone.rt2)
        fact = cholesky!(Hermitian(cone.mat2, :U), check = false)
        if isposdef(fact)
            return (logdet(fact) - cone.d * log(-u / cone.d) > eps(T))
        end
    end

    return false
end

function update_grad(cone::HypoRootdetTri)
    @assert cone.is_feas
    g = cone.grad
    ζ = cone.ζ
    cone.ϕζidi = cone.ϕ / ζ * cone.di
    ϕζidi1 = -cone.ϕζidi - 1

    g[1] = inv(ζ)
    inv_fact!(cone.Wi, cone.fact_W)
    smat_to_svec!(cone.Wi_vec, cone.Wi, cone.rt2)
    @. g[2:end] = ϕζidi1 * cone.Wi_vec

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoRootdetTri)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    Wi_vec = cone.Wi_vec
    ζ = cone.ζ
    ϕζidi = cone.ϕζidi
    c1 = -ϕζidi / ζ
    ϕζidi1 = ϕζidi + 1
    c2 = ϕζidi * (ϕζidi - cone.di)

    H[1, 1] = ζ^-2
    @. H[1, 2:end] = c1 * Wi_vec

    copytri!(cone.Wi, 'U', true)
    @views symm_kron!(H[2:end, 2:end], cone.Wi, cone.rt2)

    @inbounds for j in eachindex(Wi_vec)
        j1 = 1 + j
        Wi_vecj = c2 * Wi_vec[j]
        for i in 1:j
            H[1 + i, j1] = ϕζidi1 * H[1 + i, j1] + Wi_vec[i] * Wi_vecj
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoRootdetTri{T},
    ) where {T <: Real}
    @assert cone.grad_updated
    di = cone.di
    ζ = cone.ζ
    ϕζidi = cone.ϕζidi
    w_aux = cone.mat2
    FU = cone.fact_W.U
    ϕζidi1 = ϕζidi + 1

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        @views svec_to_smat!(w_aux, arr[2:end, j], cone.rt2)
        copytri!(w_aux, 'U', true)
        rdiv!(w_aux, FU)
        ldiv!(FU', w_aux)

        c0 = ϕζidi * tr(Hermitian(w_aux, :U))
        c1 = c0 - p / ζ
        c2 = ϕζidi * c1 - di * c0

        lmul!(ϕζidi1, w_aux)
        for i in diagind(w_aux)
            w_aux[i] += c2
        end
        rdiv!(w_aux, FU')
        ldiv!(FU, w_aux)

        prod[1, j] = c1 / -ζ
        @views smat_to_svec!(prod[2:end, j], w_aux, cone.rt2)
    end

    return prod
end

function update_inv_hess(cone::HypoRootdetTri)
    @assert cone.grad_updated
    @assert !cone.inv_hess_updated
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    @views w = cone.point[2:end]
    svec_to_smat!(cone.mat2, w, cone.rt2)
    W = Hermitian(cone.mat2, :U)
    Hi = cone.inv_hess.data
    ζ = cone.ζ
    ϕ = cone.ϕ
    di = cone.di
    ϕdi = ϕ * di
    c2 = inv(cone.ϕζidi + 1)
    c3 = ϕdi * c2 / ζ * di

    Hi[1, 1] = abs2(ζ) + ϕdi * ϕ
    @. Hi[1, 2:end] = ϕdi * w

    @views symm_kron!(Hi[2:end, 2:end], W, cone.rt2)

    @inbounds for j in eachindex(w)
        j1 = 1 + j
        scwj = c3 * w[j]
        for i in 1:j
            Hi[1 + i, j1] = c2 * Hi[1 + i, j1] + w[i] * scwj
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoRootdetTri{T},
    ) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    svec_to_smat!(cone.mat4, w, cone.rt2)
    W = Hermitian(cone.mat4, :U)
    ζ = cone.ζ
    ϕ = cone.ϕ
    di = cone.di
    ϕdi = ϕ * di
    c2 = inv(cone.ϕζidi + 1)
    c3 = c2 / ζ * di
    c4 = abs2(ζ) + ϕdi * ϕ
    w_aux = cone.mat2
    w_aux2 = cone.mat3

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]
        svec_to_smat!(w_aux, r, cone.rt2)
        copytri!(w_aux, 'U', true)

        c5 = dot(w, r)
        c6 = ϕdi * (c3 * c5 + p)
        prod[1, j] = ϕdi * c5 + c4 * p

        mul!(w_aux2, w_aux, W)
        mul!(w_aux, W, w_aux2)
        @views prod_w = prod[2:end, j]
        smat_to_svec!(prod_w, w_aux, cone.rt2)
        axpby!(c6, w, c2, prod_w)
    end

    return prod
end

function dder3(cone::HypoRootdetTri{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    dder3 = cone.dder3
    p = dir[1]
    @views r = dir[2:end]
    ζ = cone.ζ
    ϕ = cone.ϕ
    di = cone.di
    ϕζidi = cone.ϕζidi
    FU = cone.fact_W.U
    rwi = cone.mat2
    w_aux = cone.mat3
    w_aux2 = cone.mat4

    svec_to_smat!(rwi, r, cone.rt2)
    copytri!(rwi, 'U', true)
    rdiv!(rwi, FU)
    ldiv!(FU', rwi)
    c0 = tr(Hermitian(rwi, :U)) * di
    c6 = sum(abs2, rwi) * di
    ζiχ = (p - ϕ * c0) / ζ
    c1 = ζiχ^2 + ϕ / ζ * (c6 - abs2(c0)) / 2
    c7 = ϕζidi * (c1 - c6 / 2 + c0 * (ζiχ + c0 / 2))
    c8 = -ϕζidi * (ζiχ + c0)
    c9 = ϕζidi + 1

    dder3[1] = c1 / -ζ

    copyto!(w_aux2, I)
    axpby!(c9, rwi, c8, w_aux2)
    mul!(w_aux, Hermitian(rwi, :U), w_aux2)
    @inbounds for i in diagind(w_aux)
        w_aux[i] += c7
    end
    rdiv!(w_aux, FU')
    ldiv!(FU, w_aux)
    @views smat_to_svec!(dder3[2:end], w_aux, cone.rt2)

    return dder3
end
