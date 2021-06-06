"""
$(TYPEDEF)

Hypograph of perspective function of real symmetric or complex Hermitian
log-determinant cone of dimension `dim` in svec format.

    $(FUNCTIONNAME){T, R}(dim::Int, use_dual::Bool = false)
"""
mutable struct HypoPerLogdetTri{T <: Real, R <: RealOrComplex{T}} <: Cone{T}
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

    ϕ::T
    ζ::T
    ζi::T
    mat::Matrix{R}
    fact_W::Cholesky{R}
    Wi::Matrix{R}
    Wi_vec::Vector{T}
    tempw::Vector{T}
    mat2::Matrix{R}
    mat3::Matrix{R}
    mat4::Matrix{R}

    function HypoPerLogdetTri{T, R}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real, R <: RealOrComplex{T}}
        @assert dim >= 3
        cone = new{T, R}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.rt2 = sqrt(T(2))
        cone.is_complex = (R <: Complex)
        cone.d = svec_side(R, dim - 2)
        return cone
    end
end

reset_data(cone::HypoPerLogdetTri) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

function setup_extra_data!(
    cone::HypoPerLogdetTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    dim = cone.dim
    d = cone.d
    cone.mat = zeros(R, d, d)
    cone.Wi = zeros(R, d, d)
    cone.Wi_vec = zeros(T, dim - 2)
    cone.tempw = zeros(T, dim - 2)
    cone.mat2 = zeros(R, d, d)
    cone.mat3 = zeros(R, d, d)
    cone.mat4 = zeros(R, d, d)
    return cone
end

get_nu(cone::HypoPerLogdetTri) = 2 + cone.d

function set_initial_point!(
    arr::AbstractVector{T},
    cone::HypoPerLogdetTri{T, R},
    ) where {T <: Real, R <: RealOrComplex{T}}
    arr .= 0
    # central point data are the same as for hypoperlog
    (arr[1], arr[2], w) = get_central_ray_hypoperlog(cone.d)
    incr = (cone.is_complex ? 2 : 1)
    k = 3
    @inbounds for i in 1:cone.d
        arr[k] = w
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::HypoPerLogdetTri{T}) where {T <: Real}
    @assert !cone.feas_updated
    v = cone.point[2]

    if v > eps(T)
        u = cone.point[1]
        @views svec_to_smat!(cone.mat, cone.point[3:end], cone.rt2)
        fact = cone.fact_W = cholesky!(Hermitian(cone.mat, :U), check = false)
        if isposdef(fact)
            cone.ϕ = logdet(fact) - cone.d * log(v)
            cone.ζ = v * cone.ϕ - u
            cone.is_feas = (cone.ζ > eps(T))
        else
            cone.is_feas = false
        end
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoPerLogdetTri{T}) where {T <: Real}
    u = cone.dual_point[1]

    if u < -eps(T)
        v = cone.dual_point[2]
        @views svec_to_smat!(cone.mat2, cone.dual_point[3:end], cone.rt2)
        fact = cholesky!(Hermitian(cone.mat2, :U), check = false)
        if isposdef(fact)
            return (v - u * (logdet(fact) + cone.d * (1 - log(-u))) > eps(T))
        end
    end

    return false
end

function update_grad(cone::HypoPerLogdetTri)
    @assert cone.is_feas
    v = cone.point[2]
    g = cone.grad
    ζ = cone.ζ
    ζi = cone.ζi = inv(ζ)

    g[1] = ζi
    g[2] = -inv(v) - (cone.ϕ - cone.d) / ζ
    inv_fact!(cone.Wi, cone.fact_W)
    smat_to_svec!(cone.Wi_vec, cone.Wi, cone.rt2)
    # ∇ϕ = cone.Wi_vec * v
    ζvζi = -1 - v / ζ
    @. g[3:end] = ζvζi * cone.Wi_vec

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLogdetTri)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    v = cone.point[2]
    H = cone.hess.data
    d = cone.d
    ζ = cone.ζ
    ζi = cone.ζi
    σ = cone.ϕ - d
    Wi_vec = cone.Wi_vec
    Wivζi = cone.tempw
    ζiσ = σ / ζ
    vζi = v / ζ
    ζvζi = 1 + vζi
    @. Wivζi = vζi * Wi_vec

    # u, v
    H[1, 1] = abs2(ζi)
    H[1, 2] = -ζi * ζiσ
    H[2, 2] = v^-2 + abs2(ζiσ) + d / (v * ζ)

    # u, v, w
    c1 = -vζi / ζ
    @. H[1, 3:end] = c1 * Wi_vec
    c2 = (σ * vζi - 1) / ζ
    @. H[2, 3:end] = c2 * Wi_vec

    # w, w
    copytri!(cone.Wi, 'U', true)
    @views symm_kron!(H[3:end, 3:end], cone.Wi, cone.rt2)

    @inbounds for j in eachindex(Wi_vec)
        j2 = 2 + j
        Wivζij = Wivζi[j]
        for i in 1:j
            H[2 + i, j2] = ζvζi * H[2 + i, j2] + Wivζi[i] * Wivζij
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoPerLogdetTri,
    )
    @assert cone.grad_updated
    v = cone.point[2]
    d = cone.d
    ζ = cone.ζ
    w_aux = cone.mat3
    FU = cone.fact_W.U
    σ = cone.ϕ - d
    vζi1 = v / ζ + 1

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views svec_to_smat!(w_aux, arr[3:end, j], cone.rt2)
        copytri!(w_aux, 'U', true)
        rdiv!(w_aux, FU)
        ldiv!(FU', w_aux)

        qζi = q / ζ
        c0 = tr(Hermitian(w_aux, :U)) / ζ
        # ∇ϕ[r] = v * c0
        c1 = (v * c0 - p / ζ + σ * qζi) / ζ
        c3 = c1 * v - qζi
        prod[1, j] = -c1
        prod[2, j] = c1 * σ - c0 + (qζi * d + q / v) / v

        lmul!(vζi1, w_aux)
        for i in diagind(w_aux)
            w_aux[i] += c3
        end
        rdiv!(w_aux, FU')
        ldiv!(FU, w_aux)

        @views smat_to_svec!(prod[3:end, j], w_aux, cone.rt2)
    end

    return prod
end

function update_inv_hess(cone::HypoPerLogdetTri)
    @assert cone.grad_updated
    @assert !cone.inv_hess_updated
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    v = cone.point[2]
    @views w = cone.point[3:end]
    svec_to_smat!(cone.mat2, w, cone.rt2)
    W = Hermitian(cone.mat2, :U)
    Hi = cone.inv_hess.data
    d = cone.d
    ζ = cone.ζ
    ϕ = cone.ϕ
    ζv = ζ + v
    ζζvi = ζ / ζv
    c3 = v / (ζv + d * v)
    c0 = ϕ - d * ζζvi
    c2 = v * c3
    c4 = c2 * ζv
    c1 = v * ζζvi + c0 * c2

    Hi[1, 1] = abs2(v * ϕ) + ζ * (ζ + d * v) - d * abs2(ζ + v * ϕ) * c3
    Hi[1, 2] = c0 * c4
    Hi[2, 2] = c4

    @. Hi[1, 3:end] = c1 * w
    @. Hi[2, 3:end] = c2 * w

    @views Hiww = Hi[3:end, 3:end]
    symm_kron!(Hiww, W, cone.rt2)
    mul!(Hiww, w, w', c2 / ζv, ζζvi)

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoPerLogdetTri,
    )
    @assert cone.grad_updated
    v = cone.point[2]
    @views w = cone.point[3:end]
    svec_to_smat!(cone.mat4, w, cone.rt2)
    W = Hermitian(cone.mat4, :U)
    d = cone.d
    ζ = cone.ζ
    ϕ = cone.ϕ
    ζv = ζ + v
    ζζvi = ζ / ζv
    c3 = v / (ζv + d * v)
    c0 = ϕ - d * ζζvi
    c4 = v * c3 * ζv
    c6 = abs2(v * ϕ) + ζ * (ζ + d * v) - d * abs2(ζ + v * ϕ) * c3
    c7 = c4 * c0
    c8 = c7 + v * ζ
    w_aux = cone.mat2
    w_aux2 = cone.mat3

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views r = arr[3:end, j]
        svec_to_smat!(w_aux, r, cone.rt2)
        copytri!(w_aux, 'U', true)

        c1 = dot(w, r) / ζv
        c5 = c0 * p + q + c1
        c2 = v * (ζζvi * p + c3 * c5)
        prod[1, j] = c6 * p + c7 * q + c8 * c1
        prod[2, j] = c4 * c5

        mul!(w_aux2, w_aux, W)
        mul!(w_aux, W, w_aux2)
        @views prod_w = prod[3:end, j]
        smat_to_svec!(prod_w, w_aux, cone.rt2)
        axpby!(c2, w, ζζvi, prod_w)
    end

    return prod
end

function dder3(cone::HypoPerLogdetTri, dir::AbstractVector)
    @assert cone.grad_updated
    v = cone.point[2]
    dder3 = cone.dder3
    p = dir[1]
    q = dir[2]
    @views r = dir[3:end]
    d = cone.d
    ζ = cone.ζ
    FU = cone.fact_W.U
    rwi = cone.mat2
    w_aux = cone.mat3
    w_aux2 = cone.mat4
    σ = cone.ϕ - d
    viq = q / v
    viq2 = abs2(viq)
    vζi = v / ζ
    vζi1 = vζi + 1

    svec_to_smat!(rwi, r, cone.rt2)
    copytri!(rwi, 'U', true)
    rdiv!(rwi, FU)
    ldiv!(FU', rwi)
    c0 = tr(Hermitian(rwi, :U))
    c7 = sum(abs2, rwi)
    ζiχ = (-p + σ * q + c0 * v) / ζ
    c4 = (viq * (-viq * d + 2 * c0) - c7) / ζ / 2
    c1 = (abs2(ζiχ) - v * c4) / ζ
    c3 = -(ζiχ + viq) / ζ
    c5 = c3 * q + vζi * viq2
    c6 = -2 * vζi * viq - c3 * v
    c8 = c5 + c1 * v

    dder3[1] = -c1
    dder3[2] = c1 * σ + (viq2 - (d * c5 + c6 * c0 + vζi * c7)) / v - c4

    copyto!(w_aux2, I)
    axpby!(vζi1, rwi, c6, w_aux2)
    mul!(w_aux, Hermitian(rwi, :U), w_aux2)
    @inbounds for i in diagind(w_aux)
        w_aux[i] += c8
    end
    rdiv!(w_aux, FU')
    ldiv!(FU, w_aux)
    @views smat_to_svec!(dder3[3:end], w_aux, cone.rt2)

    return dder3
end
