#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

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
    mat::Matrix{R}
    fact_W::Cholesky{R}
    Wi::Matrix{R}
    Wi_vec::Vector{T}
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

function setup_extra_data!(
    cone::HypoPerLogdetTri{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    dim = cone.dim
    d = cone.d
    cone.mat = zeros(R, d, d)
    cone.Wi = zeros(R, d, d)
    cone.Wi_vec = zeros(T, dim - 2)
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
    @inbounds for i in 1:(cone.d)
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
    ζi = inv(ζ)

    g[1] = ζi
    g[2] = -inv(v) - (cone.ϕ - cone.d) / ζ
    inv_fact!(cone.Wi, cone.fact_W)
    smat_to_svec!(cone.Wi_vec, cone.Wi, cone.rt2)
    # ∇ϕ = cone.Wi_vec * v
    vζi1 = -1 - v / ζ
    @. g[3:end] = vζi1 * cone.Wi_vec

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
    Wi_vec = cone.Wi_vec
    σζi = (cone.ϕ - d) / ζ
    vζi = v / ζ

    # u, v
    H[1, 1] = ζ^-2
    H[1, 2] = -σζi / ζ
    H[2, 2] = v^-2 + abs2(σζi) + d / (ζ * v)

    # u, v, w
    vζi2 = -vζi / ζ
    c1 = ((cone.ϕ - d) * vζi - 1) / ζ
    @. H[1, 3:end] = vζi2 * Wi_vec
    @. H[2, 3:end] = c1 * Wi_vec

    # w, w
    copytri!(cone.Wi, 'U', true)
    @views symm_kron!(H[3:end, 3:end], cone.Wi, cone.rt2)

    @inbounds for j in eachindex(Wi_vec)
        j2 = 2 + j
        c2 = vζi * Wi_vec[j]
        for i in 1:j
            i2 = 2 + i
            H[i2, j2] += vζi * (H[i2, j2] + c2 * Wi_vec[i])
        end
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPerLogdetTri)
    @assert cone.grad_updated
    v = cone.point[2]
    FU = cone.fact_W.U
    d = cone.d
    ζ = cone.ζ
    σ = cone.ϕ - d
    vζi1 = (v + ζ) / ζ
    w_aux = cone.mat3

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views svec_to_smat!(w_aux, arr[3:end, j], cone.rt2)
        copytri!(w_aux, 'U', true)
        rdiv!(w_aux, FU)
        ldiv!(FU', w_aux)

        qζi = q / ζ
        c1 = tr(Hermitian(w_aux, :U)) / ζ
        # ∇ϕ[r] = v * c1
        c2 = (v * c1 - p / ζ + σ * qζi) / ζ
        c3 = c2 * v - qζi

        prod[1, j] = -c2
        prod[2, j] = c2 * σ - c1 + (qζi * d + q / v) / v

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
    c1 = v / (ζv + d * v) * v
    c2 = c1 / ζv

    # u, v
    Hi12 = Hi[1, 2] = c1 * (ζv * ϕ - d * ζ)
    Hi[1, 1] = ζζvi * d * v^2 + ζ^2 + (ϕ - ζζvi * d) * Hi12
    Hi[2, 2] = c1 * ζv

    # u, v, w
    c3 = (v * ζ + Hi12) / ζv
    @. Hi[1, 3:end] = c3 * w
    @. Hi[2, 3:end] = c1 * w

    # w, w
    @views Hiww = Hi[3:end, 3:end]
    symm_kron!(Hiww, W, cone.rt2)
    mul!(Hiww, w, w', c2, ζζvi)

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
    c1 = ζv / v + d
    w_aux = cone.mat2
    w_aux2 = cone.mat3

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views r = arr[3:end, j]
        svec_to_smat!(w_aux, r, cone.rt2)
        copytri!(w_aux, 'U', true)

        trrw = dot(w, r)
        c2 = v * (ζv * (ϕ * p + q) - d * ζ * p + trrw) / c1
        c3 = (c2 + ζ * v * p) / ζv

        prod[1, j] = ζ * ((v * (d * p * v + trrw) - d * c2) / ζv + ζ * p) + ϕ * c2
        prod[2, j] = c2

        mul!(w_aux2, w_aux, W)
        mul!(w_aux, W, w_aux2)
        @views prod_w = prod[3:end, j]
        smat_to_svec!(prod_w, w_aux, cone.rt2)
        axpby!(c3, w, ζζvi, prod_w)
    end

    return prod
end

function dder3(cone::HypoPerLogdetTri, dir::AbstractVector)
    @assert cone.grad_updated
    v = cone.point[2]
    dder3 = cone.dder3
    p = dir[1]
    q = dir[2]
    d = cone.d
    ζ = cone.ζ
    FU = cone.fact_W.U
    σ = cone.ϕ - d
    viq = q / v
    vζi1 = (v + ζ) / ζ
    rwi = cone.mat2
    w_aux = cone.mat3
    w_aux2 = cone.mat4

    @views svec_to_smat!(rwi, dir[3:end], cone.rt2)
    copytri!(rwi, 'U', true)
    rdiv!(rwi, FU)
    ldiv!(FU', rwi)

    tr1 = tr(Hermitian(rwi, :U))
    tr2 = sum(abs2, rwi)

    χ = (-p + σ * q + tr1 * v) / ζ
    c1 = (viq * (2 * tr1 - viq * d) - tr2) / (2 * ζ)
    c2 = (abs2(χ) - v * c1) / ζ
    c3 = -q * χ / ζ
    c4 = (χ * v - q) / ζ
    c5 = c3 + c2 * v

    dder3[1] = -c2
    dder3[2] = c2 * σ + (abs2(viq) - d * c3 - c4 * tr1) / v - tr2 / ζ - c1

    copyto!(w_aux2, I)
    axpby!(vζi1, rwi, c4, w_aux2)
    mul!(w_aux, Hermitian(rwi, :U), w_aux2)
    @inbounds for i in diagind(w_aux)
        w_aux[i] += c5
    end
    rdiv!(w_aux, FU')
    ldiv!(FU, w_aux)
    @views smat_to_svec!(dder3[3:end], w_aux, cone.rt2)

    return dder3
end
