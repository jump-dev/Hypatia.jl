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
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    ϕ::T
    ζ::T
    tempw::Vector{T}

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
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

function setup_extra_data!(cone::HypoPerLog{T}) where {T <: Real}
    d = cone.dim - 2
    cone.tempw = zeros(T, d)
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

    g[1] = inv(cone.ζ)
    g[2] = -(cone.ϕ - d) / ζ - inv(v)
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
    wivζi = cone.tempw
    d = length(w)
    σζi = (cone.ϕ - d) / ζ
    vζi = v / ζ
    @. wivζi = vζi / w

    # u, v
    H[1, 1] = ζ^-2
    H[1, 2] = -σζi / ζ
    H[2, 2] = v^-2 + abs2(σζi) + d / ζ / v

    # u, v, w
    vζi2 = -vζi / ζ
    c1 = ((cone.ϕ - d) * vζi - 1) / ζ
    @. H[1, 3:end] = vζi2 / w
    @. H[2, 3:end] = c1 / w

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
    d = length(w)
    σ = cone.ϕ - d
    rwi = cone.tempw
    vζi1 = v / ζ + 1

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @. @views rwi = arr[3:end, j] / w

        qζi = q / ζ
        c0 = sum(rwi) / ζ
        # ∇ϕ[r] = v * c0
        c1 = (v * c0 - p / ζ + σ * qζi) / ζ
        c3 = c1 * v - qζi
        prod[1, j] = -c1
        prod[2, j] = c1 * σ - c0 + (qζi * d + q / v) / v
        @. prod[3:end, j] = (c3 + vζi1 * rwi) / w
    end

    return prod
end

function update_inv_hess(cone::HypoPerLog)
    @assert cone.grad_updated
    @assert !cone.inv_hess_updated
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
    Hi = cone.inv_hess.data
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

    @inbounds for j in eachindex(w)
        j2 = 2 + j
        Hi[j2, j2] += abs2(w[j])
    end
    @views mul!(Hi[3:end, 3:end], w, w', c2 / ζv, ζζvi)

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::HypoPerLog,
    )
    @assert cone.grad_updated
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
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
    rw = cone.tempw

    @inbounds for j in 1:size(arr, 2)
        p = arr[1, j]
        q = arr[2, j]
        @. @views rw = arr[3:end, j] * w

        c1 = sum(rw) / ζv
        c5 = c0 * p + q + c1
        c2 = v * (ζζvi * p + c3 * c5)
        prod[1, j] = c6 * p + c7 * q + c8 * c1
        prod[2, j] = c4 * c5
        @. prod[3:end, j] = (c2 + ζζvi * rw) * w
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
    ζ = cone.ζ
    d = length(w)
    σ = cone.ϕ - d
    viq = q / v
    viq2 = abs2(viq)
    rwi = cone.tempw
    vζi = v / ζ
    vζi1 = vζi + 1

    @. @views rwi = dir[3:end] / w
    c0 = sum(rwi)
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
