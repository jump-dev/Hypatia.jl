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
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    di::T
    ϕ::T
    ζ::T
    η::T
    tempw::Vector{T}

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

function setup_extra_data!(cone::HypoGeoMean{T}) where {T <: Real}
    d = cone.dim - 1
    cone.di = inv(T(d))
    cone.tempw = zeros(T, d)
    return cone
end

get_nu(cone::HypoGeoMean) = cone.dim

function set_initial_point!(
    arr::AbstractVector{T},
    cone::HypoGeoMean{T},
    ) where {T <: Real}
    (u, w) = get_central_ray_hypogeomean(T, cone.dim - 1)
    arr[1] = u
    arr[2:end] .= w
    return arr
end

function update_feas(cone::HypoGeoMean{T}) where {T <: Real}
    @assert !cone.feas_updated
    @views w = cone.point[2:end]

    if all(>(eps(T)), w)
        cone.ϕ = exp(cone.di * sum(log, w))
        cone.ζ = cone.ϕ - cone.point[1]
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
        return (sum(log, w) - length(w) * log(-u * cone.di) > eps(T))
    end

    return false
end

function update_grad(cone::HypoGeoMean)
    @assert cone.is_feas
    @views w = cone.point[2:end]
    g = cone.grad
    ζ = cone.ζ
    cone.η = cone.ϕ / ζ * cone.di
    mθ = -1 - cone.η

    g[1] = inv(ζ)
    @. g[2:end] = mθ / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoGeoMean)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    @views w = cone.point[2:end]
    H = cone.hess.data
    ζ = cone.ζ
    η = cone.η
    c1 = η - cone.di
    c2 = η * (1 + c1) + 1

    H[1, 1] = ζ^-2

    @inbounds for j in eachindex(w)
        j1 = 1 + j
        w_j = w[j]
        c3 = η / w_j
        H[1, j1] = -c3 / ζ

        c4 = c3 * c1
        for i in 1:(j - 1)
            H[i + 1, j1] = c4 / w[i]
        end
        H[j1, j1] = c2 / w_j / w_j
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
    ζ = cone.ζ
    η = cone.η
    θ = 1 + η
    rwi = cone.tempw

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]
        @. rwi = r / w

        c1 = η * sum(rwi)
        χ = c1 - p / ζ
        ητ = η * χ - di * c1

        prod[1, j] = χ / -ζ
        @. prod[2:end, j] = (ητ + θ * rwi) / w
    end

    return prod
end

function update_inv_hess(cone::HypoGeoMean{T}) where {T <: Real}
    @assert cone.grad_updated
    @assert !cone.inv_hess_updated
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    u = cone.point[1]
    @views w = cone.point[2:end]
    Hi = cone.inv_hess.data
    ζ = cone.ζ
    ϕ = cone.ϕ
    di = cone.di
    η = cone.η
    diϕ = ϕ * di
    θi = inv(1 + η)
    c1 = θi / ζ * di

    Hi[1, 1] = abs2(ζ) + diϕ * ϕ

    @inbounds for j in eachindex(w)
        j1 = j + 1
        w_j = w[j]
        c2 = diϕ * w_j
        Hi[1, j1] = c2

        c3 = c1 * c2
        for i in 1:(j - 1)
            Hi[i + 1, j1] = c3 * w[i]
        end
        Hi[j1, j1] = (c3 + θi * w_j) * w_j
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::HypoGeoMean{T},
    ) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    ζ = cone.ζ
    ϕ = cone.ϕ
    di = cone.di
    η = cone.η
    diϕ = ϕ * di
    θi = inv(1 + η)
    c1 = di * η * θi
    c2 = abs2(ζ) + diϕ * ϕ
    rw = cone.tempw

    @inbounds for j in 1:size(prod, 2)
        p = arr[1, j]
        @views r = arr[2:end, j]
        @. rw = r * w

        c3 = dot(w, r)
        c4 = diϕ * p + c1 * c3

        prod[1, j] = c2 * p + diϕ * c3
        @. prod[2:end, j] = (c4 + θi * rw) * w
    end

    return prod
end

function dder3(cone::HypoGeoMean{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    @views w = cone.point[2:end]
    dder3 = cone.dder3
    p = dir[1]
    @views r = dir[2:end]
    ζ = cone.ζ
    di = cone.di
    ϕ = cone.ϕ
    η = cone.η
    θ = 1 + η
    rwi = cone.tempw

    @. rwi = r / w

    tr1 = sum(rwi)
    χ = -p / ζ + η * tr1
    ητ = η * (χ - di * tr1)
    ηυh = η * (sum(abs2, rwi) - di * abs2(tr1)) / 2
    c1 = χ * ητ + (η - di) * ηυh

    dder3[1] = (abs2(χ) + ηυh) / -ζ
    @. dder3[2:end] = (c1 + rwi * (ητ + θ * rwi)) / w

    return dder3
end

function get_central_ray_hypogeomean(::Type{T}, d::Int) where {T <: Real}
    c = sqrt(T(d * (5 * d + 2) + 1))
    u = -sqrt((-c + 3 * d + 1) / T(2 + 2 * d))
    w = -u * (d + 1 + c) / T(2 * d)
    return (u, w)
end
