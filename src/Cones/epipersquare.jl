"""
$(TYPEDEF)

Epigraph of perspective function of halved squared Euclidean norm (AKA rotated
second-order) cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""
mutable struct EpiPerSquare{T <: Real} <: Cone{T}
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
    sqrt_hess_prod_updated::Bool
    inv_sqrt_hess_prod_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    dist::T
    rtdist::T
    denom::T
    sqrt_hess_vec::Vector{T}
    inv_sqrt_hess_vec::Vector{T}

    function EpiPerSquare{T}(dim::Int) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.dim = dim
        return cone
    end
end

use_dual_barrier(::EpiPerSquare) = false

reset_data(cone::EpiPerSquare) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.sqrt_hess_prod_updated =
    cone.inv_sqrt_hess_prod_updated = false)

use_sqrt_hess_oracles(::Int, cone::EpiPerSquare) = true

get_nu(cone::EpiPerSquare) = 2

function set_initial_point!(arr::AbstractVector, cone::EpiPerSquare)
    @views arr[1:2] .= 1
    @views arr[3:end] .= 0
    return arr
end

# TODO refac with dual feas check
function update_feas(cone::EpiPerSquare{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]

    if u > eps(T) && v > eps(T)
        @views w = cone.point[3:end]
        cone.dist = u * v - sum(abs2, w) / 2
        cone.is_feas = (cone.dist > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiPerSquare{T}) where T
    u = cone.dual_point[1]
    v = cone.dual_point[2]

    if u > eps(T) && v > eps(T)
        @views w = cone.dual_point[3:end]
        return (u * v - sum(abs2, w) / 2 > eps(T))
    end

    return false
end

function update_grad(cone::EpiPerSquare)
    @assert cone.is_feas

    @. cone.grad = cone.point / cone.dist
    g2 = cone.grad[2]
    cone.grad[2] = -cone.grad[1]
    cone.grad[1] = -g2

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiPerSquare)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)
    H = cone.hess.data

    mul!(H, cone.grad, cone.grad')
    inv_dist = inv(cone.dist)
    @inbounds for j in 3:cone.dim
        H[j, j] += inv_dist
    end
    H[1, 2] -= inv_dist

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiPerSquare)
    @assert cone.is_feas
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    Hi = cone.inv_hess.data

    mul!(Hi, cone.point, cone.point')
    @inbounds for j in 3:cone.dim
        Hi[j, j] += cone.dist
    end
    Hi[1, 2] -= cone.dist

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiPerSquare,
    )
    u = cone.point[1]
    v = cone.point[2]
    w = @view cone.point[3:end]

    @inbounds for j in 1:size(prod, 2)
        uj = arr[1, j]
        vj = arr[2, j]
        @views wj = arr[3:end, j]
        ga = (dot(w, wj) - v * uj - u * vj) / cone.dist
        prod[1, j] = -ga * v - vj
        prod[2, j] = -ga * u - uj
        @. @views prod[3:end, j] = ga * w + wj
    end
    @. prod /= cone.dist

    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiPerSquare,
    )
    @assert cone.is_feas

    @inbounds @views for j in 1:size(prod, 2)
        pa = dot(cone.point, arr[:, j])
        @. prod[:, j] = pa * cone.point
    end
    @. @views prod[1, :] -= cone.dist * arr[2, :]
    @. @views prod[2, :] -= cone.dist * arr[1, :]
    @. @views prod[3:end, :] += cone.dist * arr[3:end, :]

    return prod
end

function update_sqrt_hess_prod(cone::EpiPerSquare{T}) where T
    @assert cone.is_feas
    @assert !cone.sqrt_hess_prod_updated
    if !isdefined(cone, :sqrt_hess_vec)
        cone.sqrt_hess_vec = zeros(T, cone.dim)
    end

    rtdist = cone.rtdist = sqrt(cone.dist)
    cone.denom = 2 * rtdist + cone.point[1] + cone.point[2]
    vec = cone.sqrt_hess_vec
    @. @views vec[3:end] = cone.point[3:end] / rtdist
    vec[1] = -cone.point[2] / rtdist - 1
    vec[2] = -cone.point[1] / rtdist - 1

    cone.sqrt_hess_prod_updated = true
    return
end

function update_inv_sqrt_hess_prod(cone::EpiPerSquare{T}) where T
    @assert cone.is_feas
    @assert !cone.inv_sqrt_hess_prod_updated
    if !isdefined(cone, :inv_sqrt_hess_vec)
        cone.inv_sqrt_hess_vec = zeros(T, cone.dim)
    end

    rtdist = cone.rtdist = sqrt(cone.dist)
    cone.denom = 2 * rtdist + cone.point[1] + cone.point[2]
    vec = cone.inv_sqrt_hess_vec
    copyto!(vec, cone.point)
    vec[1:2] .+= rtdist

    cone.inv_sqrt_hess_prod_updated = true
    return
end

function sqrt_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSquare{T},
    ) where {T <: Real}
    if !cone.sqrt_hess_prod_updated
        update_sqrt_hess_prod(cone)
    end
    vec = cone.sqrt_hess_vec
    rtdist = cone.rtdist

    @inbounds @views for j in 1:size(arr, 2)
        dotj = dot(vec, arr[:, j]) / cone.denom
        @. prod[:, j] = dotj * vec
    end
    @. @views prod[1, :] -= arr[2, :] / rtdist
    @. @views prod[2, :] -= arr[1, :] / rtdist
    @. @views prod[3:end, :] += arr[3:end, :] / rtdist

    return prod
end

function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiPerSquare{T},
    ) where {T <: Real}
    if !cone.inv_sqrt_hess_prod_updated
        update_inv_sqrt_hess_prod(cone)
    end
    vec = cone.inv_sqrt_hess_vec
    rtdist = cone.rtdist

    @inbounds @views for j in 1:size(arr, 2)
        dotj = dot(vec, arr[:, j]) / cone.denom
        @. prod[:, j] = dotj * vec
    end
    @. @views prod[1, :] -= arr[2, :] * rtdist
    @. @views prod[2, :] -= arr[1, :] * rtdist
    @. @views prod[3:end, :] += arr[3:end, :] * rtdist

    return prod
end

function dder3(cone::EpiPerSquare, dir::AbstractVector)
    @assert cone.grad_updated
    dim = cone.dim
    dder3 = cone.dder3
    point = cone.point
    u = point[1]
    v = point[2]
    u_dir = dir[1]
    v_dir = dir[2]
    @views w = point[3:end]
    @views w_dir = dir[3:end]

    jdotpd = u * v_dir + v * u_dir - dot(w, w_dir)
    hess_prod!(dder3, dir, cone)
    dotdHd = -dot(dir, dder3)
    dotpHd = dot(point, dder3)
    dder3 .*= jdotpd
    @. @views dder3[3:end] += dotdHd * w + dotpHd * w_dir
    dder3[1] += -dotdHd * v - dotpHd * v_dir
    dder3[2] += -dotdHd * u - dotpHd * u_dir
    dder3 ./= 2 * cone.dist

    return dder3
end
