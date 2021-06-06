"""
$(TYPEDEF)

Epigraph of Euclidean norm (AKA second-order) cone of dimension `dim`.

    $(FUNCTIONNAME){T}(dim::Int)
"""
mutable struct EpiNormEucl{T <: Real} <: Cone{T}
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

    dist::T

    function EpiNormEucl{T}(dim::Int) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.dim = dim
        return cone
    end
end

use_dual_barrier(::EpiNormEucl) = false

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = false)

use_sqrt_hess_oracles(::Int, cone::EpiNormEucl) = true

get_nu(cone::EpiNormEucl) = 2

function set_initial_point!(
    arr::AbstractVector,
    cone::EpiNormEucl{T},
    ) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

# TODO refac with dual feas check
function update_feas(cone::EpiNormEucl{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > eps(T)
        @views w = cone.point[2:end]
        cone.dist = (abs2(u) - sum(abs2, w)) / 2
        cone.is_feas = (cone.dist > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiNormEucl{T}) where T
    u = cone.dual_point[1]

    if u > eps(T)
        w = view(cone.dual_point, 2:cone.dim)
        @views dual_dist = abs2(u) - sum(abs2, cone.dual_point[2:end])
        return (dual_dist > 2 * eps(T))
    end

    return false
end

function update_grad(cone::EpiNormEucl)
    @assert cone.is_feas

    @. cone.grad = cone.point / cone.dist
    cone.grad[1] *= -1

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiNormEucl)
    @assert cone.grad_updated
    isdefined(cone, :hess) || alloc_hess!(cone)

    mul!(cone.hess.data, cone.grad, cone.grad')
    inv_dist = inv(cone.dist)
    @inbounds for j in eachindex(cone.grad)
        cone.hess[j, j] += inv_dist
    end
    cone.hess[1, 1] -= inv_dist + inv_dist

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiNormEucl)
    @assert cone.is_feas
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)

    mul!(cone.inv_hess.data, cone.point, cone.point')
    @inbounds for j in eachindex(cone.grad)
        cone.inv_hess[j, j] += cone.dist
    end
    cone.inv_hess[1, 1] -= cone.dist + cone.dist

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormEucl,
    )
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    @inbounds for j in 1:size(prod, 2)
        uj = arr[1, j]
        wj = @view arr[2:end, j]
        ga = (dot(w, wj) - u * uj) / cone.dist
        prod[1, j] = -ga * u - uj
        @. @views prod[2:end, j] = ga * w + wj
    end
    @. prod ./= cone.dist

    return prod
end

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::EpiNormEucl,
    )
    @assert cone.is_feas

    @inbounds for j in 1:size(prod, 2)
        @views pa = dot(cone.point, arr[:, j])
        @. @views prod[:, j] = pa * cone.point
    end
    @. @views prod[1, :] -= cone.dist * arr[1, :]
    @. @views prod[2:end, :] += cone.dist * arr[2:end, :]

    return prod
end

function sqrt_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormEucl{T},
    ) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    rt2 = sqrt(T(2))
    distrt2 = cone.dist * rt2
    rtdist = sqrt(cone.dist)
    urtdist = u + rtdist * rt2
    @inbounds for j in 1:size(arr, 2)
        uj = arr[1, j]
        @views wj = arr[2:end, j]
        dotwwj = dot(w, wj)
        prod[1, j] = (u * uj - dotwwj) / distrt2
        wmulj = (dotwwj / urtdist - uj) / distrt2
        @. @views prod[2:end, j] = w * wmulj + wj / rtdist
    end

    return prod
end

function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat{T},
    arr::AbstractVecOrMat{T},
    cone::EpiNormEucl{T},
    ) where {T <: Real}
    @assert cone.is_feas
    u = cone.point[1]
    w = @view cone.point[2:end]

    rt2 = sqrt(T(2))
    rtdist = sqrt(cone.dist)
    urtdist = u + rtdist * rt2
    @inbounds for j in 1:size(arr, 2)
        uj = arr[1, j]
        @views wj = arr[2:end, j]
        dotwwj = dot(w, wj)
        prod[1, j] = (u * uj + dotwwj) / rt2
        wmulj = (dotwwj / urtdist + uj) / rt2
        @. @views prod[2:end, j] = w * wmulj + wj * rtdist
    end

    return prod
end

function dder3(cone::EpiNormEucl, dir::AbstractVector)
    @assert cone.grad_updated
    dim = cone.dim
    dder3 = cone.dder3
    point = cone.point
    u = point[1]
    u_dir = dir[1]
    @views w = point[2:end]
    @views w_dir = dir[2:end]

    jdotpd = u * u_dir - dot(w, w_dir)
    hess_prod!(dder3, dir, cone)
    dotdHd = -dot(dir, dder3)
    dotpHd = dot(point, dder3)
    dder3 .*= jdotpd
    @. @views dder3[2:end] += dotdHd * w + dotpHd * w_dir
    dder3[1] += -dotdHd * u - dotpHd * u_dir
    dder3 ./= 2 * cone.dist

    return dder3
end
