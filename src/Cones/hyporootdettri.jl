#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

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
    η::T
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
    d = cone.d
    arr .= 0
    # central point data are the same as for hypogeomean
    (arr[1], w) = get_central_ray_hypogeomean(T, d)
    incr = (cone.is_complex ? 2 : 1)
    k = 2
    @inbounds for i in 1:d
        arr[k] = w
        k += incr * i + 1
    end
    return arr
end

function update_feas(cone::HypoRootdetTri{T}) where {T <: Real}
    @assert !cone.feas_updated
    @views svec_to_smat!(cone.mat, cone.point[2:end], cone.rt2)

    fact = cone.fact_W = cholesky!(Hermitian(cone.mat, :U), check = false)
    if isposdef(fact)
        cone.ϕ = exp(cone.di * logdet(fact))
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
            return (exp(cone.di * logdet(fact)) + cone.di * u > eps(T))
        end
    end

    return false
end

function update_grad(cone::HypoRootdetTri)
    @assert cone.is_feas
    g = cone.grad
    ζ = cone.ζ
    cone.η = cone.ϕ / ζ * cone.di
    mθ = -1 - cone.η

    g[1] = inv(ζ)
    inv_fact!(cone.Wi, cone.fact_W)
    smat_to_svec!(cone.Wi_vec, cone.Wi, cone.rt2)
    @. g[2:end] = mθ * cone.Wi_vec

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoRootdetTri)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    Wi_vec = cone.Wi_vec
    ζ = cone.ζ
    η = cone.η
    θ = 1 + η
    c1 = -η / ζ
    c2 = η * (η - cone.di)

    H[1, 1] = ζ^-2
    @. H[1, 2:end] = c1 * Wi_vec

    copytri!(cone.Wi, 'U', true)
    @views symm_kron!(H[2:end, 2:end], cone.Wi, cone.rt2)

    @inbounds for j in eachindex(Wi_vec)
        j1 = 1 + j
        Wi_vecj = c2 * Wi_vec[j]
        for i in 1:j
            H[1 + i, j1] = θ * H[1 + i, j1] + Wi_vec[i] * Wi_vecj
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
    η = cone.η
    w_aux = cone.mat2
    FU = cone.fact_W.U
    θ = 1 + η

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        @views svec_to_smat!(w_aux, arr[2:end, j], cone.rt2)
        copytri!(w_aux, 'U', true)
        rdiv!(w_aux, FU)
        ldiv!(FU', w_aux)

        c1 = η * tr(Hermitian(w_aux, :U))
        χ = c1 - p / ζ
        ητ = η * χ - di * c1

        prod[1, j] = χ / -ζ
        lmul!(θ, w_aux)
        for i in diagind(w_aux)
            w_aux[i] += ητ
        end
        rdiv!(w_aux, FU')
        ldiv!(FU, w_aux)
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
    η = cone.η
    diϕ = ϕ * di
    θi = inv(1 + η)
    c1 = di * η * θi

    Hi[1, 1] = abs2(ζ) + diϕ * ϕ
    @. Hi[1, 2:end] = diϕ * w

    @views symm_kron!(Hi[2:end, 2:end], W, cone.rt2)

    @inbounds for j in eachindex(w)
        j1 = 1 + j
        scwj = c1 * w[j]
        for i in 1:j
            Hi[1 + i, j1] = θi * Hi[1 + i, j1] + w[i] * scwj
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
    η = cone.η
    θ = 1 + η
    θi = inv(θ)
    c1 = η / θ
    w_aux = cone.mat2
    w_aux2 = cone.mat3

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]
        svec_to_smat!(w_aux, r, cone.rt2)
        copytri!(w_aux, 'U', true)

        c2 = di * dot(w, r)
        c3 = ϕ * p * di
        c4 = c3 + c1 * c2

        prod[1, j] = ζ * p * ζ + ϕ * (c3 + c2)
        mul!(w_aux2, w_aux, W)
        mul!(w_aux, W, w_aux2)
        @views prod_w = prod[2:end, j]
        smat_to_svec!(prod_w, w_aux, cone.rt2)
        axpby!(c4, w, θi, prod_w)
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
    η = cone.η
    FU = cone.fact_W.U
    rwi = cone.mat2
    w_aux = cone.mat3
    w_aux2 = cone.mat4

    svec_to_smat!(rwi, r, cone.rt2)
    copytri!(rwi, 'U', true)
    rdiv!(rwi, FU)
    ldiv!(FU', rwi)

    tr1 = tr(Hermitian(rwi, :U))
    χ = -p / ζ + η * tr1
    ητ = η * (χ - di * tr1)
    ηυh = η * (sum(abs2, rwi) - di * abs2(tr1)) / 2
    c1 = χ * ητ + (η - di) * ηυh

    dder3[1] = (abs2(χ) + ηυh) / -ζ

    copyto!(w_aux2, I)
    axpby!(1 + η, rwi, ητ, w_aux2)
    mul!(w_aux, Hermitian(rwi, :U), w_aux2)
    @inbounds for i in diagind(w_aux)
        w_aux[i] += c1
    end
    rdiv!(w_aux, FU')
    ldiv!(FU, w_aux)
    @views smat_to_svec!(dder3[2:end], w_aux, cone.rt2)

    return dder3
end
