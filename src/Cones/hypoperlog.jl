"""
$(TYPEDEF)

Hypograph of perspective function of sum-log cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int, use_dual::Bool = false)
"""
mutable struct HypoPerLog{T <: Real} <: Cone{T}
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
    inv_hess_aux_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    α::T
    ϕ::T
    ζ::T
    ζi::T
    σ::T
    c0::T
    c4::T
    Hiuu::T
    tempw1::Vector{T}

    function HypoPerLog{T}(
        dim::Int;
        use_dual::Bool = false,
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        return cone
    end
end

reset_data(cone::HypoPerLog) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_aux_updated =
    cone.hess_fact_updated = false)

use_sqrt_hess_oracles(cone::HypoPerLog) = false

function setup_extra_data!(cone::HypoPerLog{T}) where {T <: Real}
    d = cone.dim - 2
    cone.tempw1 = zeros(T, d)
    return cone
end

get_nu(cone::HypoPerLog) = cone.dim

function set_initial_point!(arr::AbstractVector, cone::HypoPerLog)
    (arr[1], arr[2], w) = get_central_ray_hypoperlog(cone.dim - 2)
    @views arr[3:end] .= w
    return arr
end

function update_feas(cone::HypoPerLog{T}) where {T <: Real}
    @assert !cone.feas_updated
    v = cone.point[2]
    @views w = cone.point[3:end]

    if v > eps(T) && all(>(eps(T)), w)
        u = cone.point[1]
        # cone.ϕ = sum(log, w) - length(w) * log(v)
        cone.ϕ = sum(log(wi / v) for wi in w)
        cone.ζ = v * cone.ϕ - u
        cone.is_feas = (cone.ζ > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoPerLog{T}) where {T <: Real}
    u = cone.dual_point[1]
    @views w = cone.dual_point[3:end]

    if all(>(eps(T)), w) && u < -eps(T)
        v = cone.dual_point[2]
        sumlog = sum(log(w_i / -u) for w_i in w)
        return (v - u * (sumlog + length(w)) > eps(T))
    end

    return false
end

function update_grad(cone::HypoPerLog)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
    ζ = cone.ζ
    g = cone.grad

    cone.σ = cone.ϕ - d
    ζi = cone.ζi = inv(cone.ζ)
    g[1] = ζi
    g[2] = -cone.σ / ζ - inv(v)
    vζi1 = -1 - v / ζ
    @. g[3:end] = vζi1 / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLog)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    v = cone.point[2]
    @views w = cone.point[3:end]
    H = cone.hess.data
    g = cone.grad
    ζ = cone.ζ
    ζi = cone.ζi
    wivζi = cone.tempw1
    d = length(w)

    σζi = cone.σ / ζ
    vζi = v / ζ
    @. wivζi = vζi / w

    # u, v
    H[1, 1] = ζ^-2
    H[1, 2] = -σζi / ζ
    H[2, 2] = v^-2 + abs2(σζi) + d / ζ / v

    # u, v, w
    vζi2 = -vζi / ζ
    c1 = ((cone.ϕ - d) * vζi - 1) / ζ
    @inbounds @. begin
        H[1, 3:end] = vζi2 / w
        H[2, 3:end] = c1 / w
    end

    # w
    @inbounds for j in eachindex(wivζi)
        j2 = 2 + j
        wivζij = wivζi[j]
        for i in 1:j
            H[2 + i, j2] = wivζi[i] * wivζij
        end
        H[j2, j2] -= g[j2] / w[j]
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoPerLog,
    )
    v = cone.point[2]
    @views w = cone.point[3:end]
    ζ = cone.ζ
    r = cone.tempw1
    σ = cone.σ
    d = length(w)
    ζi = cone.ζi
    rwi = cone.tempw1
    vζi1 = v / ζ + 1

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @. @views r = arr[3:end, j]

        @. rwi = r / w
        c0 = sum(rwi)
        # ∇ϕ[r] = v * c0
        c1 = (v * c0 - p + σ * q) / ζ / ζ
        c3 = c1 * v - q / ζ
        prod[1, j] = -c1
        prod[2, j] = c1 * σ + (q * d / v - c0) / ζ + q / v / v
        @. prod[3:end, j] = (c3 + vζi1 * rwi) / w
    end

    return prod
end

function update_inv_hess_aux(cone::HypoPerLog)
    @assert cone.feas_updated
    @assert !cone.inv_hess_aux_updated
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
    ζ = cone.ζ
    ζv = ζ + v
    ζuζ = 2 * ζ + u
    den = ζv + d * v

    c0 = cone.ϕ - d * ζ / (ζ + v)
    Hiuu = abs2(ζ + u) + ζ * (den - v) - d * abs2(ζuζ) * v / den
    c4 = v^2 / den * ζv
    cone.c0 = c0
    cone.c4 = c4
    cone.Hiuu = Hiuu
    # α is a scaling of w
    cone.α = v * ζ / ζv

    cone.inv_hess_aux_updated = true
    return
end

function update_inv_hess(cone::HypoPerLog)
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    v = cone.point[2]
    @views w = cone.point[3:end]
    Hi = cone.inv_hess.data
    ζ = cone.ζ
    ζv = ζ + v
    ζζv = ζ / ζv
    c0 = cone.c0
    c4 = cone.c4

    Hi[1, 1] = cone.Hiuu
    Hi[1, 2] = c0 * c4
    Hi[2, 2] = c4

    γ_vec = cone.tempw1
    @. γ_vec = w / ζv
    @inbounds begin
        @. Hi[1, 3:end] = (cone.α + c0 * c4 / ζv) * w
        @. Hi[2, 3:end] = c4 * w / ζv
        @views mul!(Hi[3:end, 3:end], γ_vec, γ_vec', c4, true)
        for j in eachindex(w)
            j2 = 2 + j
            Hi[j2, j2] += ζζv * abs2(w[j])
        end
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoPerLog,
    )
    cone.inv_hess_aux_updated || update_inv_hess_aux(cone)
    v = cone.point[2]
    @views w = cone.point[3:end]
    ζ = cone.ζ
    ζv = ζ + v
    ζζvi = ζ / ζv
    ζi = inv(ζ)
    c0 = cone.c0
    c4 = cone.c4
    α = cone.α

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @views r = arr[3:end, j]

        c1 = dot(r, w)
        qγr = q + c1 / ζv
        cv = c4 * (c0 * p + qγr)
        c2 = p * α + cv / ζv
        prod[1, j] = cone.Hiuu * p + c4 * c0 * qγr + c1 * α
        prod[2, j] = cv
        @. @views prod[3:end, j] = (c2 + ζζvi * w * r) * w
    end

    return prod
end

function dder3(cone::HypoPerLog{T}, dir::AbstractVector{T}) where {T <: Real}
    @assert cone.grad_updated
    v = cone.point[2]
    @views w = cone.point[3:end]
    dder3 = cone.dder3
    p = dir[1]
    q = dir[2]
    @views r = dir[3:end]
    ζ = cone.ζ
    σ = cone.σ
    d = length(r)
    viq = q / v
    viq2 = abs2(viq)
    rwi = cone.tempw1
    ζi = cone.ζi
    vζi = v / ζ
    vζi1 = vζi + 1

    @. rwi = r / w
    c0 = sum(rwi)
    c7 = sum(abs2, rwi)
    ζiχ = (-p + σ * q + c0 * v) / ζ
    c4 = (viq * (-viq * d + 2 * c0) - c7) / ζ / 2
    c1 = (abs2(ζiχ) - v * c4) / ζ
    c2 = (viq * d - c0) / ζ
    c3 = -ζi * (ζiχ + viq)
    c5 = c3 * q + vζi * viq2
    c6 = -2 * vζi * viq - c3 * v
    c8 = c5 + c1 * v

    dder3[1] = -c1
    dder3[2] = c1 * σ + (viq2 - (d * c5 + c6 * c0 + vζi * c7)) / v - c4
    @. dder3[3:end] = (c8 + rwi * (c6 + vζi1 * rwi)) / w

    return dder3
end

# see analysis in
# https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
function get_central_ray_hypoperlog(d::Int)
    if d <= 10
        # lookup points where x = f'(x)
        return central_rays_hypoperlog[d, :]
    end
    # use nonlinear fit for higher dimensions
    x = inv(d)
    if d <= 70
        u = 4.657876 * x ^ 2 - 3.116192 * x + 0.000647
        v = 0.424682 * x + 0.553392
        w = 0.760412 * x + 1.001795
    else
        u = -3.011166 * x - 0.000122
        v = 0.395308 * x + 0.553955
        w = 0.837545 * x + 1.000024
    end
    return [u, v, w]
end

const central_rays_hypoperlog = [
    -0.827838387  0.805102007  1.290927686
    -0.689607388  0.724605082  1.224617936
    -0.584372665  0.68128058  1.182421942
    -0.503499342  0.65448622  1.153053152
    -0.440285893  0.636444224  1.131466926
    -0.389979809  0.623569352  1.114979519
    -0.349255921  0.613978276  1.102013921
    -0.315769104  0.606589839  1.091577908
    -0.287837744  0.600745284  1.083013
    -0.264242734  0.596019009  1.075868782
    ]
