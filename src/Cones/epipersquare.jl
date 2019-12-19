#=
Copyright 2018, Chris Coey and contributors

epigraph of perspective of (half) square function (AKA rotated second-order cone)
(u in R, v in R_+, w in R^n) : u >= v*1/2*norm_2(w/v)^2
note v*1/2*norm_2(w/v)^2 = 1/2*sum_i(w_i^2)/v

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(2*u*v - norm_2(w)^2)
=#

mutable struct EpiPerSquare{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    dist::T

    function EpiPerSquare{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiPerSquare{T}(dim::Int) where {T <: Real} = EpiPerSquare{T}(dim, false)

reset_data(cone::EpiPerSquare) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function setup_data(cone::EpiPerSquare{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

get_nu(cone::EpiPerSquare) = 2

function set_initial_point(arr::AbstractVector, cone::EpiPerSquare)
    arr[1:2] .= 1
    arr[3:end] .= 0
    return arr
end

function update_feas(cone::EpiPerSquare)
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]

    if u > 0 && v > 0
        w = view(cone.point, 3:cone.dim)
        cone.dist = u * v - sum(abs2, w) / 2
        cone.is_feas = (cone.dist > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
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
    H = cone.hess.data

    mul!(H, cone.grad, cone.grad')
    invdist = inv(cone.dist)
    @inbounds for j in 3:cone.dim
        H[j, j] += invdist
    end
    H[1, 2] -= invdist

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiPerSquare)
    @assert cone.is_feas

    mul!(cone.inv_hess.data, cone.point, cone.point')
    @inbounds for j in 3:cone.dim
        cone.inv_hess.data[j, j] += cone.dist
    end
    cone.inv_hess.data[1, 2] -= cone.dist

    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_hess_prod(cone::EpiPerSquare) = nothing
update_inv_hess_prod(cone::EpiPerSquare) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSquare)
    u = cone.point[1]
    v = cone.point[2]
    w = @view cone.point[3:end]

    @inbounds for j in 1:size(prod, 2)
        uj = arr[1, j]
        vj = arr[2, j]
        wj = @view arr[3:end, j]
        ga = (dot(w, wj) - v * uj - u * vj) / cone.dist
        prod[1, j] = -ga * v - vj
        prod[2, j] = -ga * u - uj
        @. prod[3:end, j] = ga * w + wj
    end
    @. prod /= cone.dist

    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSquare)
    @assert cone.is_feas

    @inbounds for j in 1:size(prod, 2)
        @views pa = dot(cone.point, arr[:, j])
        @. prod[:, j] = pa * cone.point
    end
    @. @views prod[1, :] -= cone.dist * arr[2, :]
    @. @views prod[2, :] -= cone.dist * arr[1, :]
    @. @views prod[3:end, :] += cone.dist * arr[3:end, :]

    return prod
end

# TODO fix
function hess_sqrt_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSquare{T}) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    w = @view cone.point[3:end]

    rt2 = sqrt(T(2))
    distrt2 = cone.dist * rt2
    rtdist = sqrt(cone.dist)
    urtdist = u + rtdist * rt2
    @inbounds for j in 1:size(arr, 2)
        uj = arr[1, j]
        vj = arr[2, j]
        @views wj = arr[3:end, j]
        dotwwj = dot(w, wj)
        prod[1, j] = (v * vj - dotwwj) / distrt2
        prod[2, j] = (u * uj - dotwwj) / distrt2
        wmulj = (dotwwj / urtdist - (uj + vj) / 2) / distrt2
        @. prod[3:end, j] = w * wmulj + wj / rtdist
    end

    return prod
end

# TODO fix
function inv_hess_sqrt_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiPerSquare{T}) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    w = @view cone.point[3:end]

    rt2 = sqrt(T(2))
    rtdist = sqrt(cone.dist)
    urtdist = u + rtdist * rt2
    @inbounds for j in 1:size(arr, 2)
        uj = arr[1, j]
        vj = arr[2, j]
        @views wj = arr[3:end, j]
        dotwwj = dot(w, wj)
        prod[1, j] = (v * vj + dotwwj) / rt2
        prod[2, j] = (u * uj + dotwwj) / rt2
        wmulj = (dotwwj / urtdist + uj + vj) / rt2
        @. prod[3:end, j] = w * wmulj + wj * rtdist
    end

    return prod
end
