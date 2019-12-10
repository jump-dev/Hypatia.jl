#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd, halved
-log(u^2 - norm_2(w)^2) / 2

contains some adapted code from https://github.com/embotech/ecos
TODO all oracles assumed we are using dense version of the Nesterov Todd scaling
TODO factor out repetitions in products, hessians, inverse hessians
TODO probably undo (a-b)(a+b) dist calculations
TODO can get rid of jdot if don't like it
=#

mutable struct EpiNormEucl{T <: Real} <: Cone{T}
    use_scaling::Bool
    use_3order_corr::Bool
    try_scaled_updates::Bool # run algorithm in scaled variables for numerical reasons TODO decide whether to keep this as an option
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}
    scaled_point::Vector{T} # TODO not in other cones, including in this one because suspecting step max dist issues without using this

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    scaling_updated::Bool # TODO remove
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    normalized_point::Vector{T} # TODO can this and normalized_dual_point be removed?
    normalized_dual_point::Vector{T}
    nt_point::Vector{T} # actually a normalized NT point
    nt_point_sqrt::Vector{T}

    dist::T
    dual_dist::T
    rt_dist_ratio::T # sqrt(dist / dual_dist)
    rt_rt_dist_ratio::T # sqrt(sqrt(dist / dual_dist))
    correction::Vector{T}

    function EpiNormEucl{T}(
        dim::Int;
        use_scaling::Bool = true,
        use_3order_corr::Bool = true,
        try_scaled_updates::Bool = false,
        ) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.dim = dim
        cone.use_scaling = use_scaling
        cone.use_3order_corr = use_3order_corr
        cone.try_scaled_updates = try_scaled_updates
        return cone
    end
end

use_dual(cone::EpiNormEucl) = false # self-dual

use_scaling(cone::EpiNormEucl) = cone.use_scaling

use_3order_corr(cone::EpiNormEucl) = cone.use_3order_corr

try_scaled_updates(cone::EpiNormEucl) = cone.try_scaled_updates

load_dual_point(cone::EpiNormEucl, dual_point::AbstractVector) = copyto!(cone.dual_point, dual_point)

reset_data(cone::EpiNormEucl) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO only allocate the fields we use
function setup_data(cone::EpiNormEucl{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.scaled_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.normalized_point = zeros(T, dim)
    cone.normalized_dual_point = zeros(T, dim)
    cone.nt_point = zeros(T, dim)
    cone.nt_point_sqrt = zeros(T, dim)
    cone.correction = zeros(T, dim)
    # initial values for scaling related fields, using fact that initial point is identity
    if cone.use_scaling
        cone.nt_point[1] = 1
        cone.nt_point_sqrt[1] = 1
        cone.normalized_point[1] = 1
        cone.normalized_dual_point[1] = 1
        cone.scaled_point[1] = 1
        cone.dual_dist = 1
        cone.rt_dist_ratio = 1
        cone.rt_rt_dist_ratio = 1
    end
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

function update_hess(cone::EpiNormEucl)
    @assert cone.grad_updated
    @assert cone.is_feas

    if cone.use_scaling
        # analogous to W as a function of nt_point_sqrt
        # H = (2 * J * nt_point * nt_point' * J - J) * constant
        mul!(cone.hess.data, cone.nt_point, cone.nt_point', 2, false)
        cone.hess.data[:, 1] *= -1
        cone.hess += I # TODO inefficient
        cone.hess.data[1, :] *= -1
        cone.hess.data ./= cone.rt_dist_ratio
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
        # Hinv = (2 * nt_point * nt_point' - J) * constant
        mul!(cone.inv_hess.data, cone.nt_point, cone.nt_point', 2, false)
        cone.inv_hess += I # TODO inefficient
        cone.inv_hess[1, 1] -= 2
        cone.inv_hess.data .*= cone.rt_dist_ratio
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
        hyperbolic_householder(prod, arr, cone.nt_point, cone.rt_dist_ratio, Winv = true)
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
        hyperbolic_householder(prod, arr, cone.nt_point, cone.rt_dist_ratio, Winv = false)
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
        hyperbolic_householder(prod, arr, cone.nt_point_sqrt, cone.rt_rt_dist_ratio, Winv = true)
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

    jdot_p_s = jdot(point, primal_dir)
    @. corr = jdot_p_s * dual_dir
    dot_s_z = dot(primal_dir, dual_dir)
    dot_p_z = dot(point, dual_dir)
    corr[1] += dot_s_z * point[1] - dot_p_z * primal_dir[1]
    @. @views corr[2:end] += -dot_s_z * point[2:end] + dot_p_z * primal_dir[2:end]
    corr ./= cone.dist

    return corr
end

# TODO try using scaled_point instead of primal and dual points
function step_max_dist(cone::EpiNormEucl{T}, primal_dir::AbstractVector{T}, dual_dir::AbstractVector{T}) where {T}
    if cone.try_scaled_updates
        # these were not calculated
        cone.normalized_point .= cone.point ./ sqrt(cone.dist)
        cone.normalized_dual_point .= cone.dual_point ./ sqrt(cone.dual_dist)
    end
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
    @views point_dir_dist = jdot(point, dir)
    fact = (point_dir_dist + dir[1]) / (point[1] + 1)
    dist2n = zero(T)
    for i in 2:length(point)
        dist2n += abs2(dir[i] - fact * point[i])
    end
    return -point_dir_dist + sqrt(dist2n)
end

function step_and_update_scaling(cone::EpiNormEucl{T}, primal_dir::AbstractVector, dual_dir::AbstractVector, step_size::T) where {T}
    @assert cone.feas_updated
    nt_point = cone.nt_point
    nt_point_sqrt = cone.nt_point_sqrt
    normalized_point = cone.normalized_point
    normalized_dual_point = cone.normalized_dual_point

    # TODO put this somehwere more appropriate, better to judge when scaled updates are decided upon
    @views cone.dual_dist = abs2(cone.dual_point[1]) - sum(abs2, cone.dual_point[2:end])
    @assert cone.dual_dist >= 0

    if cone.try_scaled_updates
        # based on CVXOPT code
        # scale the primal and dual directions under old scaling, store results as points to be normalized
        hyperbolic_householder(normalized_point, primal_dir, nt_point_sqrt, cone.rt_rt_dist_ratio, Winv = true)
        hyperbolic_householder(normalized_dual_point, dual_dir, nt_point_sqrt, cone.rt_rt_dist_ratio)
        # get new primal/dual points but in the old scaling
        scaled_point = cone.scaled_point
        @. normalized_point *= step_size
        @. normalized_dual_point *= step_size
        @. normalized_point += scaled_point
        @. normalized_dual_point += scaled_point

        # normalize
        primal_dir_dist = distnorm(normalized_point)
        dual_dir_dist = distnorm(normalized_dual_point)
        normalized_point ./= primal_dir_dist
        normalized_dual_point ./= dual_dir_dist

        gamma = sqrt((1 + dot(normalized_point, normalized_dual_point)) / 2)
        vs = dot(nt_point_sqrt, normalized_point)
        vz = jdot(nt_point_sqrt, normalized_dual_point)
        vq = (vs + vz) / 2 / gamma
        vu = vs - vz
        scaled_point[1] = gamma

        w1 = 2 * nt_point_sqrt[1] * vq - (normalized_point[1] + normalized_dual_point[1] ) / 2 / gamma
        d = (nt_point_sqrt[1] * vu - (normalized_point[1] - normalized_dual_point[1]) / 2) / (w1 + 1)

        # updates the scaled point
        @views copyto!(scaled_point[2:end], nt_point_sqrt[2:end])
        @. scaled_point[2:end] *= (vu - 2 * d * vq)
        @. scaled_point[2:end] += normalized_point[2:end] * (1 - d / gamma) / 2
        @. scaled_point[2:end] += normalized_dual_point[2:end] * (1 + d / gamma) / 2
        @. scaled_point *= sqrt(primal_dir_dist * dual_dir_dist)

        # updates the NT scaling point
        @. nt_point = 2 * vq * nt_point_sqrt
        nt_point[1] -= normalized_point[1] / 2 / gamma
        @. @views nt_point[2:end] += normalized_point[2:end] / 2 / gamma
        @. nt_point -= normalized_dual_point / gamma / 2

        # updates the squareroot of the NT scaling point
        copyto!(nt_point_sqrt, nt_point)
        nt_point_sqrt[1] += 1
        @. nt_point_sqrt /= sqrt(2 * nt_point_sqrt[1])

        cone.rt_dist_ratio *= primal_dir_dist / dual_dir_dist
        cone.rt_rt_dist_ratio *= sqrt(primal_dir_dist / dual_dir_dist)
    else
        # section 4 CVXOPT paper / part of ECOS' update each iteration
        normalized_point .= cone.point ./ sqrt(cone.dist)
        normalized_dual_point .= cone.dual_point ./ sqrt(cone.dual_dist)
        gamma = sqrt((1 + dot(normalized_point, normalized_dual_point)) / 2)

        nt_point[1] = normalized_point[1] + normalized_dual_point[1]
        @. @views nt_point[2:end] = normalized_point[2:end] - normalized_dual_point[2:end]
        nt_point ./= 2 * gamma

        copyto!(nt_point_sqrt, nt_point)
        nt_point_sqrt[1] += 1
        nt_point_sqrt ./= sqrt(2 * nt_point_sqrt[1])

        cone.rt_dist_ratio = sqrt(cone.dist / cone.dual_dist)
        cone.rt_rt_dist_ratio = sqrt(cone.rt_dist_ratio)
    end

    return
end

# calculates fact * (2vv' - J) * X if Winv = false
# calculates 1/fact * (2Jvv'J - J) * X if Winv = true
# function hyperbolic_householder(prod::AbstractVecOrMat, arr::AbstractVecOrMat, v::AbstractVector, fact::Real; Winv::Bool = false)
#     if !Winv
#         v[2:end] *= -1
#     end
#     @inbounds for j in 1:size(arr, 2)
#         @views pa = jdot(arr[:, j], v)
#         @. @views prod[:, j] = -2 * v * pa
#     end
#     @. prod += arr
#     @. @views prod[1, :] *= -1
#     if Winv
#         prod ./= fact
#     else
#         v[2:end] *= -1
#         prod .*= fact
#     end
#     return prod
# end
function hyperbolic_householder(prod::AbstractVecOrMat, arr::AbstractVecOrMat, v::AbstractVector, fact::Real; Winv::Bool = false)
    if Winv
        v[2:end] *= -1
    end
    for j in 1:size(prod, 2)
        @views pa = 2 * dot(v, arr[:, j])
        @. @views prod[:, j] = pa * v
    end
    @. prod[1, :] -= arr[1, :]
    @. prod[2:end, :] += arr[2:end, :]
    if Winv
        prod ./= fact
        v[2:end] *= -1
    else
        prod .*= fact
    end
    return prod
end

function distnorm(x::AbstractVector)
    x1 = x[1]
    @views x2 = norm(x[2:end])
    return sqrt(x1 - x2) * sqrt(x1 + x2)
end

jdot(x::AbstractVector, y::AbstractVector) = @views x[1] * y[1] - dot(x[2:end], y[2:end])
