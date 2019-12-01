#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd, halved
-log(u^2 - norm_2(w)^2) / 2

contains some adapted code from https://github.com/embotech/ecos
TODO all oracles assumed we are using dense version of the Nesterov Todd scaling
TODO factor out repetitions in products, hessians, inverse hessians
TODO factor out eta
TODO probably undo (a-b)(a+b) dist calculations
=#

mutable struct EpiNormEucl{T <: Real} <: Cone{T}
    use_scaling::Bool
    use_3order_corr::Bool
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    scaling_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    normalized_point::Vector{T} # TODO can this and normalized_dual_point be removed?
    normalized_dual_point::Vector{T}
    w::Vector{T} # TODO rename - confusing with w part of point
    # experimental
    lambda::Vector{T} # TODO think about naming, it's w_bar in cvxopt paper
    # TODO improve naming of short 1 letter variable names below
    v::Vector{T}

    dist::T
    dual_dist::T
    gamma::T
    ww::T
    b::T
    c::T
    correction::Vector{T}

    function EpiNormEucl{T}(
        dim::Int;
        use_scaling::Bool = true,
        use_3order_corr::Bool = true,
        ) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.dim = dim
        cone.use_scaling = use_scaling
        cone.use_3order_corr = use_3order_corr
        return cone
    end
end

use_dual(cone::EpiNormEucl) = false # self-dual

use_scaling(cone::EpiNormEucl) = cone.use_scaling

use_3order_corr(cone::EpiNormEucl) = cone.use_3order_corr

# try_scaled_updates(cone::EpiNormEucl) = cone.try_scaled_updates # TODO

load_dual_point(cone::EpiNormEucl, dual_point::AbstractVector) = copyto!(cone.dual_point, dual_point)

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.scaling_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormEucl{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.normalized_point = zeros(T, dim)
    cone.normalized_dual_point = zeros(T, dim)
    cone.w = zeros(T, dim)
    cone.v = zeros(T, dim)
    cone.correction = zeros(T, dim)
    return
end

get_nu(cone::EpiNormEucl) = 1

function set_initial_point(arr::AbstractVector, cone::EpiNormEucl)
    arr .= 0
    arr[1] = 1
    return arr
end

function update_feas(cone::EpiNormEucl)
    @assert !cone.feas_updated

    u = cone.point[1]
    if u > 0
        @views cone.dist = abs2(u) - sum(abs2, cone.point[2:end])
        cone.is_feas = (cone.dist > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormEucl)
    @assert cone.is_feas

    @. cone.grad = cone.point / cone.dist
    cone.grad[1] *= -1

    cone.grad_updated = true
    return cone.grad
end

function update_scaling(cone::EpiNormEucl)
    @assert !cone.scaling_updated
    @assert cone.feas_updated
    w = cone.w
    v = cone.v
    normalized_point = cone.normalized_point
    normalized_dual_point = cone.normalized_dual_point

    @views cone.dual_dist = abs2(cone.dual_point[1]) - sum(abs2, cone.dual_point[2:end])
    @assert cone.dual_dist >= 0
    normalized_point .= cone.point ./ sqrt(cone.dist)
    normalized_dual_point .= cone.dual_point ./ sqrt(cone.dual_dist)
    cone.gamma = sqrt((1 + dot(normalized_point, normalized_dual_point)) / 2)

    w[1] = normalized_point[1] + normalized_dual_point[1]
    @. @views w[2:end] = normalized_point[2:end] - normalized_dual_point[2:end]
    w ./= 2 * cone.gamma
    w1 = w[1]

    copyto!(v, w)
    v[1] += 1
    v ./= sqrt(2 * (w1 + 1))

    @views w2nw2n = sum(abs2, w[2:end])
    cone.ww = abs2(w1) + w2nw2n
    w11 = 1 + w1
    w2nw2ndiv = w2nw2n / w11
    cone.b = 1 + (2 + w2nw2ndiv) / w11
    cone.c = w11 + w2nw2ndiv

    cone.scaling_updated = true
    return cone.scaling_updated
end

function update_hess(cone::EpiNormEucl)
    @assert cone.grad_updated
    @assert cone.is_feas

    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w
        hess = cone.hess.data
        hess[1, 1] = cone.ww
        @. @views hess[1, 2:end] = -cone.c * w[2:end]
        @views mul!(hess[2:end, 2:end], w[2:end], w[2:end]', cone.b, false)
        hess[2:end, 2:end] += I # TODO inefficient
        hess .*= sqrt(cone.dual_dist / cone.dist)
    else
        mul!(cone.hess.data, cone.grad, cone.grad', 2, false)
        cone.hess += inv(cone.dist) * I # TODO inefficient
        cone.hess[1, 1] -= 2 / cone.dist
    end

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::EpiNormEucl)
    @assert cone.is_feas

    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w
        inv_hess = cone.inv_hess.data
        inv_hess[1, 1] = cone.ww
        @. @views inv_hess[1, 2:end] = cone.c * w[2:end]
        @views mul!(inv_hess[2:end, 2:end], w[2:end], w[2:end]', cone.b, false)
        inv_hess[2:end, 2:end] += I # TODO inefficient
        inv_hess .*= sqrt(cone.dist / cone.dual_dist)
    else
        mul!(cone.inv_hess.data, cone.point, cone.point', 2, false)
        cone.inv_hess += cone.dist * I
        cone.inv_hess[1, 1] -= 2 * cone.dist
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

update_hess_prod(cone::EpiNormEucl) = nothing
update_inv_hess_prod(cone::EpiNormEucl) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas

    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w
        @views prod[1, :] = cone.ww * arr[1, :] - cone.c * arr[2:end, :]' * w[2:end]
        for j in 1:size(prod, 2)
            @views pa = dot(w[2:end], arr[2:end, j])
            @. @views prod[2:end, j] = pa * w[2:end] * cone.b
        end
        @. @views prod[2:end, :] += arr[2:end, :]
        @. @views prod[2:end, :] -= cone.c * arr[1, :]' * w[2:end]
        prod .*= sqrt(cone.dual_dist / cone.dist)
    else
        p1 = cone.point[1]
        @views p2 = cone.point[2:end]
        @inbounds for j in 1:size(prod, 2)
            arr_1j = arr[1, j]
            @views arr_2j = arr[2:end, j]
            ga = 2 * (dot(p2, arr_2j) - p1 * arr_1j) / cone.dist
            prod[1, j] = -ga * p1 - arr_1j
            @. prod[2:end, j] = ga * p2 + arr_2j
        end
        @. prod ./= cone.dist
    end

    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas

    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        w = cone.w
        @views prod[1, :] = cone.ww * arr[1, :] + cone.c * arr[2:end, :]' * w[2:end]
        for j in 1:size(prod, 2)
            @views pa = dot(w[2:end], arr[2:end, j])
            @. @views prod[2:end, j] = pa * w[2:end] * cone.b
        end
        @. @views prod[2:end, :] += arr[2:end, :]
        @. @views prod[2:end, :] += cone.c * arr[1, :]' * w[2:end]
        prod .*= sqrt(cone.dist / cone.dual_dist)
    else
        @inbounds for j in 1:size(prod, 2)
            @views pa = 2 * dot(cone.point, arr[:, j])
            @. prod[:, j] = pa * cone.point
        end
        @. @views prod[1, :] -= cone.dist * arr[1, :]
        @. @views prod[2:end, :] += cone.dist * arr[2:end, :]
    end

    return prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormEucl)
    @assert cone.is_feas

    if cone.use_scaling
        if !cone.scaling_updated
            update_scaling(cone)
        end
        v = cone.v
        vu = v[1]
        @views vw = v[2:end]
        divdist = sqrt(sqrt(cone.dual_dist) / sqrt(cone.dist))
        @inbounds for j in 1:size(arr, 2)
            uj = arr[1, j]
            @views wj = arr[2:end, j]
            vmul = 2 * (uj * vu - dot(wj, vw)) * divdist
            prod[1, j] = vmul * vu - uj * divdist
            @. prod[2:end, j] = -vmul * vw + wj * divdist
        end
    else
        u = cone.point[1]
        w = view(cone.point, 2:cone.dim)
        rtdist = sqrt(cone.dist)
        urtdist = u + rtdist
        @inbounds for j in 1:size(arr, 2)
            uj = arr[1, j]
            @views wj = arr[2:end, j]
            dotwwj = dot(w, wj)
            prod[1, j] = (u * uj - dotwwj) / cone.dist
            wmulj = (dotwwj / urtdist - uj) / cone.dist
            @. prod[2:end, j] = w * wmulj + wj / rtdist
        end
    end

    return prod
end

# TODO cleanup
function correction(cone::EpiNormEucl, primal_dir::AbstractVector, dual_dir::AbstractVector)
    @assert cone.grad_updated
    corr = cone.correction
    point = cone.point

    @views jdot_p_s = point[1] * primal_dir[1] - dot(point[2:end], primal_dir[2:end])
    @. corr = jdot_p_s * dual_dir
    dot_s_z = dot(primal_dir, dual_dir)
    dot_p_z = dot(point, dual_dir)
    corr[1] += dot_s_z * point[1] - dot_p_z * primal_dir[1]
    @. @views corr[2:end] += -dot_s_z * point[2:end] + dot_p_z * primal_dir[2:end]
    corr ./= cone.dist

    return corr
end

function step_max_dist(cone::EpiNormEucl{T}, primal_dir::AbstractVector{T}, dual_dir::AbstractVector{T}) where {T}
    @assert cone.scaling_updated

    primal_step_dist = sqrt(cone.dist) / dist_to_bndry(cone, cone.normalized_point, primal_dir)
    dual_step_dist = sqrt(cone.dual_dist) / dist_to_bndry(cone, cone.normalized_dual_point, dual_dir)
    step_dist = T(Inf)
    if primal_step_dist > 0
        step_dist = min(step_dist, primal_step_dist)
    end
    if dual_step_dist > 0
        step_dist = min(step_dist, dual_step_dist)
    end

    return step_dist
end

function dist_to_bndry(::EpiNormEucl{T}, point::Vector{T}, dir::AbstractVector{T}) where {T}
    @views point_dir_dist = point[1] * dir[1] - dot(point[2:end], dir[2:end])
    fact = (point_dir_dist + dir[1]) / (point[1] + 1)
    dist2n = zero(T)
    for i in 2:length(point)
        dist2n += abs2(dir[i] - fact * point[i])
    end
    return -point_dir_dist + sqrt(dist2n)
end
