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
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    di::T
    ϕ::T
    ζ::T
    ζi::T
    tempw1::Vector{T}
    tempw2::Vector{T}

    function HypoGeoMean{T}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data!(cone::HypoGeoMean{T}) where {T <: Real}
    d = cone.dim - 1
    cone.tempw1 = zeros(T, d)
    cone.tempw2 = zeros(T, d)
    cone.di = inv(T(d))
    return cone
end

get_nu(cone::HypoGeoMean) = cone.dim

function set_initial_point!(arr::AbstractVector{T}, cone::HypoGeoMean{T}) where T
    d = cone.dim - 1
    c = sqrt(T(5 * d ^ 2 + 2 * d + 1))
    arr[1] = -sqrt((-c + 3 * d + 1) / T(2 * cone.dim))
    @views arr[2:end] .= (c - d + 1) / sqrt(cone.dim * (-2 * c + 6 * d + 2))
    return arr
end

function update_feas(cone::HypoGeoMean{T}) where T
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

function is_dual_feas(cone::HypoGeoMean{T}) where T
    u = cone.dual_point[1]
    @views w = cone.dual_point[2:end]

    if u < -eps(T) && all(>(eps(T)), w)
        return ((cone.dim - 1) * exp(cone.di * sum(log, w)) + u > eps(T))
    end

    return false
end

function update_grad(cone::HypoGeoMean)
    @assert cone.is_feas
    u = cone.point[1]
    @views w = cone.point[2:end]

    ζi = cone.ζi = inv(cone.ζ)
    cone.grad[1] = ζi
    gconst = -cone.di * cone.ϕ * ζi - 1
    @. @views cone.grad[2:end] = gconst / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoGeoMean)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data
    u = cone.point[1]
    @views w = cone.point[2:end]
    ζ = cone.ζ
    ζi = cone.ζi
    di = cone.di
    ϕz = di * cone.ϕ / ζ
    ϕzm1 = ϕz - di
    constww = ϕz * (1 + ϕzm1) + 1

    H[1, 1] = abs2(ζi)
    @inbounds for j in eachindex(w)
        j1 = j + 1
        wj = w[j]
        ϕzwj = ϕz / wj
        H[1, j1] = -ϕzwj / ζ
        ϕzwj2 = ϕzwj * ϕzm1
        @inbounds for i in 1:(j - 1)
            H[i + 1, j1] = ϕzwj2 / w[i]
        end
        H[j1, j1] = constww / wj / wj
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoGeoMean{T},
    ) where T
    @assert cone.grad_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    ζ = cone.ζ
    ζi = cone.ζi
    di = cone.di
    ϕz = di * cone.ϕ / ζ
    ϕzm1 = ϕz - di
    constww = ϕz + 1

    @inbounds @views for j in 1:size(arr, 2)
        p = arr[1, j]
        pζ = p / ζ
        prod_w = prod[2:end, j]
        @. prod_w = arr[2:end, j] / w
        c0 = sum(prod_w)
        prod[1, j] = (pζ - ϕz * c0) * ζi
        dot1 = ϕz * (-pζ + ϕzm1 * c0)
        @. prod_w = (dot1 + constww * prod_w) / w
    end

    return prod
end

function update_inv_hess(cone::HypoGeoMean{T}) where T
    @assert !cone.inv_hess_updated
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    Hi = cone.inv_hess.data
    u = cone.point[1]
    @views w = cone.point[2:end]
    d = length(w)
    ϕid = cone.ϕ * cone.di
    denom = cone.dim * cone.ϕ - d * u
    zd2 = d * cone.ζ / denom

    Hi[1, 1] = cone.ϕ * (cone.dim * ϕid - 2 * u) + abs2(u)
    @inbounds for j in eachindex(w)
        j1 = j + 1
        wj = w[j]
        ϕwj = ϕid * wj
        Hi[1, j1] = ϕwj
        ϕwjd = ϕwj / denom
        @inbounds for i in 1:(j - 1)
            Hi[i + 1, j1] = ϕwjd * w[i]
        end
        Hi[j1, j1] = (ϕwjd + zd2 * wj) * wj
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoGeoMean{T},
    ) where T
    u = cone.point[1]
    @views w = cone.point[2:end]
    d = length(w)
    ϕ = cone.ϕ
    ϕid = ϕ * cone.di
    const1 = ϕ * (cone.dim * ϕid - 2 * u) + abs2(u)
    denom = cone.dim * ϕ - d * u
    zd2 = d * cone.ζ / denom

    @inbounds @views for j in 1:size(prod, 2)
        p = arr[1, j]
        prod_w = prod[2:end, j]
        @. prod_w = w * arr[2:end, j]
        dot1 = sum(prod_w) * ϕid
        prod[1, j] = dot1 + const1 * p
        dot2 = dot1 / denom + p * ϕid
        @. prod_w = (dot2 + zd2 * prod_w) * w
    end

    return prod
end

function dder3(cone::HypoGeoMean, dir::AbstractVector)
    @assert cone.grad_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    dder3 = cone.dder3
    di = cone.di
    p = dir[1]
    @views r = dir[2:end]
    ϕ = cone.ϕ
    ζ = cone.ζ
    ζi = cone.ζi

    rwi = cone.tempw1
    @. rwi = r / w
    c0 = sum(rwi) * di

    ξb = cone.tempw2
    # ∇2h[r] = ϕ * (c0 - rwi) / w * di
    @. ξb = ζi * ϕ * (c0 - rwi) / w * di
    ζiχ = ζi * (p - ϕ * c0)
    ξbξ = dot(ξb, r) / 2
    c1 = -ζi * (ζiχ^2 - ξbξ)

    c2 = -ζi / 2
    rwi2 = sum(abs2, rwi) * di
    w_aux = ξb
    w_aux .*= ζiχ
    # add c2 * ∇3h[r, r]
    @. w_aux += c2 * ϕ * ((c0 -  rwi)^2 - rwi2 + rwi^2) * di / w

    dder3[1] = c1
    @. dder3[2:end] = (abs2(rwi) - c1 * ϕ * di) / w - w_aux

    return dder3
end
