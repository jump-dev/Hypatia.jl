"""
$(TYPEDEF)

Hypograph of geometric mean cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int, use_dual::Bool = false)
"""
mutable struct HypoGeoMean{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int

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
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    di::T
    ϕ::T
    ζ::T
    ζi::T
    ϕζidi::T
    tempw1::Vector{T}

    function HypoGeoMean{T}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        return cone
    end
end

reset_data(cone::HypoGeoMean) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = false)

use_sqrt_hess_oracles(cone::HypoGeoMean) = false

function setup_extra_data!(cone::HypoGeoMean{T}) where {T <: Real}
    d = cone.dim - 1
    cone.di = inv(T(d))
    cone.tempw1 = zeros(T, d)
    return cone
end

get_nu(cone::HypoGeoMean) = cone.dim

function set_initial_point!(
    arr::AbstractVector{T},
    cone::HypoGeoMean{T},
    ) where {T <: Real}
    d = cone.dim - 1
    c = sqrt(T(5 * d ^ 2 + 2 * d + 1))
    arr[1] = -sqrt((-c + 3 * d + 1) / T(2 * cone.dim))
    arr[2:end] .= (c - d + 1) / sqrt(cone.dim * (-2 * c + 6 * d + 2))
    return arr
end

function update_feas(cone::HypoGeoMean{T}) where {T <: Real}
    @assert !cone.feas_updated
    u = cone.point[1]
    @views w = cone.point[2:end]

    if all(>(eps(T)), w)
        cone.ϕ = exp(cone.di * sum(log, w))
        cone.ζ = cone.ϕ - u
        cone.is_feas = (cone.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoGeoMean{T}) where {T <: Real}
    u = cone.dual_point[1]
    @views w = cone.dual_point[2:end]

    if (u < -eps(T)) && all(>(eps(T)), w)
        return (length(w) * exp(cone.di * sum(log, w)) + u > eps(T))
    end

    return false
end

function update_grad(cone::HypoGeoMean)
    @assert cone.is_feas
    @views w = cone.point[2:end]
    g = cone.grad
    ζi = cone.ζi = inv(cone.ζ)
    cone.ϕζidi = cone.ϕ * ζi * cone.di

    g[1] = ζi
    ϕζidi1 = -cone.ϕζidi - 1
    @. g[2:end] = ϕζidi1 / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoGeoMean)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    @views w = cone.point[2:end]
    H = cone.hess.data
    ζi = cone.ζi
    ϕζidi = cone.ϕζidi
    ϕzm1 = ϕζidi - cone.di
    c1 = ϕζidi * (1 + ϕzm1) + 1

    H[1, 1] = abs2(ζi)

    @inbounds for j in eachindex(w)
        j1 = j + 1
        wj = w[j]
        ϕzwj = ϕζidi / wj
        H[1, j1] = -ϕzwj * ζi

        ϕzwj2 = ϕzwj * ϕzm1
        for i in 1:(j - 1)
            H[i + 1, j1] = ϕzwj2 / w[i]
        end
        H[j1, j1] = c1 / wj / wj
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoGeoMean{T},
    ) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    di = cone.di
    ζi = cone.ζi
    ϕζidi = cone.ϕζidi
    rwi = cone.tempw1
    ϕζidi1 = ϕζidi + 1

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]
        @. rwi = r / w
        c0 = ϕζidi * sum(rwi)
        c1 = c0 - ζi * p
        prod[1, j] = -ζi * c1
        # ∇2h[r] = ϕ * (c0 - rwi) / w * di
        c2 = ϕζidi * c1 - di * c0
        @. prod[2:end, j] = (c2 + ϕζidi1 * rwi) / w
    end

    return prod
end

function update_inv_hess(cone::HypoGeoMean{T}) where {T <: Real}
    @assert !cone.inv_hess_updated
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    u = cone.point[1]
    @views w = cone.point[2:end]
    Hi = cone.inv_hess.data
    ϕ = cone.ϕ
    di = cone.di
    ϕdi = ϕ * di
    c2 = inv(cone.ϕζidi + 1)
    c3 = c2 * cone.ζi * di

    Hi[1, 1] = abs2(cone.ζ) + ϕdi * ϕ

    @inbounds for j in eachindex(w)
        j1 = j + 1
        wj = w[j]
        ϕwj = ϕdi * wj
        Hi[1, j1] = ϕwj

        ϕwjd = c3 * ϕwj
        for i in 1:(j - 1)
            Hi[i + 1, j1] = ϕwjd * w[i]
        end
        Hi[j1, j1] = (ϕwjd + c2 * wj) * wj
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoGeoMean{T},
    ) where {T <: Real}
    @views w = cone.point[2:end]
    ϕ = cone.ϕ
    di = cone.di
    ϕdi = ϕ * di
    rw = cone.tempw1
    c2 = inv(cone.ϕζidi + 1)
    c3 = c2 * cone.ζi * di
    c4 = abs2(cone.ζ) + ϕdi * ϕ

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]
        @. rw = r * w
        c5 = sum(rw)
        prod[1, j] = ϕdi * c5 + c4 * p
        c6 = ϕdi * (c3 * c5 + p)
        @. prod[2:end, j] = (c6 + c2 * rw) * w
    end

    return prod
end

function dder3(cone::HypoGeoMean{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    dder3 = cone.dder3
    p = dir[1]
    @views r = dir[2:end]
    di = cone.di
    ϕ = cone.ϕ
    ζi = cone.ζi
    ϕζidi = cone.ϕζidi
    rwi = cone.tempw1

    @. rwi = r / w
    c5 = ϕζidi / 2
    c0 = sum(rwi) * di
    c6 = sum(abs2, rwi) * di
    ζiχ = ζi * (p - ϕ * c0)
    c1 = ζiχ^2 + ζi * ϕ * (c6 - abs2(c0)) / 2
    c2 = ϕζidi * (c1 - c6 / 2)
    c4 = ϕζidi * ζiχ
    c7 = c2 + c0 * (c4 + c5 * c0)
    c8 = -c4 - 2 * c5 * c0
    c9 = 2 * c5 + 1

    dder3[1] = -ζi * c1
    @. dder3[2:end] = (c7 + rwi * (c8 + c9 * rwi)) / w

    return dder3
end
