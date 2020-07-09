#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of sum of logarithms
(u in R, v in R_+, w in R_+^d) : u <= v*sum(log.(w/v))

barrier modified from "Primal-Dual Interior-Point Methods for Domain-Driven Formulations" by Karimi & Tuncel, 2019
-log(sum_i v*log(w_i/v) - u) - sum_i log(w_i) - d*log(v)
=#

mutable struct HypoPerLog{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    point::Vector{T}
    dual_point::Vector{T}
    timer::TimerOutput

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    correction::Vector{T}
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    lwv::T
    vlwvu::T
    lvwnivlwvu::T
    vwivlwvu::Vector{T}

    function HypoPerLog{T}(
        dim::Int;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

# TODO only allocate the fields we use
function setup_data(cone::HypoPerLog{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.correction = zeros(T, dim)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.vwivlwvu = zeros(T, dim - 2)
    return
end

get_nu(cone::HypoPerLog) = 1 + 2 * (cone.dim - 2)

function set_initial_point(arr::AbstractVector, cone::HypoPerLog)
    (arr[1], arr[2], w) = get_central_ray_hypoperlog(cone.dim - 2)
    arr[3:end] .= w
    return arr
end

function update_feas(cone::HypoPerLog{T}) where {T}
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]

    if v > eps(T) && all(>(eps(T)), w)
        cone.lwv = sum(log(wi / v) for wi in w)
        cone.vlwvu = v * cone.lwv - u
        cone.is_feas = (cone.vlwvu > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

# TODO higher dim case
is_dual_feas(cone::HypoPerLog) = true
# function is_dual_feas(cone::HypoPerLog{T}) where {T}
#     @assert cone.dim == 3
#     u = cone.dual_point[1]
#     v = cone.dual_point[2]
#     @views w = cone.dual_point[3:end]
#     if u < -eps(T) && all(>(eps(T)), w)
#         return (v - u - u * log(-w / u) > eps(T))
#     end
#     return false
# end

function update_grad(cone::HypoPerLog)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    d = length(w)
    g = cone.grad

    g[1] = inv(cone.vlwvu)
    cone.lvwnivlwvu = (d - cone.lwv) / cone.vlwvu
    g[2] = cone.lvwnivlwvu - d / v
    gden = -1 - v / cone.vlwvu
    @. g[3:end] = gden / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLog)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    d = length(w)
    vwivlwvu = cone.vwivlwvu
    lvwnivlwvu = cone.lvwnivlwvu
    g = cone.grad
    H = cone.hess.data

    vivlwvu = v / cone.vlwvu
    @. vwivlwvu = vivlwvu / w
    H[1, 1] = abs2(g[1])
    H[1, 2] = lvwnivlwvu / cone.vlwvu
    @. H[1, 3:end] = -vwivlwvu / cone.vlwvu
    H[2, 2] = abs2(lvwnivlwvu) + d * (g[1] + inv(v)) / v
    hden = (-v * lvwnivlwvu - 1) / cone.vlwvu
    @. H[2, 3:end] = hden / w
    @inbounds for j in 1:d
        j2 = 2 + j
        @inbounds for i in 1:j
            H[2 + i, j2] = vwivlwvu[i] * vwivlwvu[j]
        end
        H[j2, j2] -= g[j2] / w[j]
    end

    cone.hess_updated = true
    return cone.hess
end

# TODO simplify and remove allocations
function correction(
    cone::HypoPerLog{T},
    primal_dir::AbstractVector{T},
    ) where {T <: Real}

    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    u_dir = primal_dir[1]
    v_dir = primal_dir[2]
    w_dir = view(primal_dir, 3:cone.dim)

    w_dim = length(w)
    z = v * sum(log(wi / v) for wi in w) - u
    sumlogw = sum(log.(w))
    dzdv = -w_dim * log(v) - w_dim + sumlogw

    corr = cone.correction
    s1_sqr = abs2(u_dir)
    s2_sqr = abs2(v_dir)
    w_wi = sum(w_dir[i] / w[i] for i in eachindex(w))
    w_wi_sqr = abs2(w_wi)
    w_sqr_wi_sqr = sum(w_dir[i] / w[i] * w_dir[i] / w[i] for i in eachindex(w))

    const1 = abs2(u_dir - dzdv * v_dir)
    const2 = v * w_sqr_wi_sqr
    const3 = 2 * v * w_wi_sqr / z
    const4 = v_dir * dzdv
    const5 = 2 * (const4 - u_dir)
    const6 = dzdv * v / z
    const7 = 2 * dzdv - w_dim
    const8 = ((u_dir - dzdv * v_dir) * v / z + v_dir) / z

    corr[1] = ((
        const1 +
        w_wi * v * (const5 + v * w_wi)
        ) * 2 / z +
        -2 * w_wi * v_dir +
        w_dim * s2_sqr / v +
        const2) / z / z

    corr[2] =
        -2 * dzdv * (const1 + w_wi_sqr * abs2(v) + w_wi * v * const5) / z ^ 3 +
        -w_dim * v_dir * (const5 + const4) / v / z / z  +
        2 * w_wi * (v_dir * const7 - u_dir) / z / z +
        -abs2(v_dir) * w_dim * (1 / z + 2 / v) / v / v +
        (const3 + w_sqr_wi_sqr - const2 * dzdv / z) / z

    @views corr[3:end] .= (
        -2 * v * const1 / z .+
        -2 * u_dir * v_dir .+
        s2_sqr * const7) ./ w / z / z +
        (2 * v * w_wi / z .+ w_dir ./ w) * const8 * 2 ./ w +
        (-const3 .- 2 * w_wi * w_dir ./ w .- w_sqr_wi_sqr) * abs2(v) / z / z ./ w +
            -(v / z + 1) * 2 * w_dir .* w_dir ./ w ./ w ./ w

    corr ./= -2

    return corr
end

# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
function get_central_ray_hypoperlog(w_dim::Int)
    if w_dim <= 10
        # lookup points where x = f'(x)
        return central_rays_hypoperlog[w_dim, :]
    end
    # use nonlinear fit for higher dimensions
    x = inv(w_dim)
    if w_dim <= 70
        u = -1.974777 * x ^ 2 + 0.413520 * x - 0.706751
        v = -1.213389 * x + 1.413551
        w = -0.406380 * x + 1.411894
    else
        u = 0.405290 * x - 0.707011
        v = -1.238597 * x + 1.414216
        w = -0.511055 * x + 1.414163
    end
    return [u, v, w]
end

const central_rays_hypoperlog = [
    -0.827838399  0.805102005  1.290927713;
    -0.751337431  0.980713381  1.317894791;
    -0.716423551  1.079796942  1.331762729;
    -0.699644766  1.144036715 1.341797042;
    -0.69134357  1.188706149  1.349742329;
    -0.687251501  1.221310686  1.3562255;
    -0.685353717  1.246016352  1.361602711;
    -0.684641818  1.265307905  1.366119586;
    -0.684585293  1.280747581  1.369956554;
    -0.684893372  1.293360445  1.373249434;
    ]

# TODO add hess prod, inv hess etc functions
# NOTE old EpiPerExp code below may be useful (cone vector is reversed)

# function update_feas(cone::EpiPerExp)
#     @assert !cone.feas_updated
#     (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
#
#     if u > 0 && v > 0
#         cone.luv = log(u / v)
#         cone.vluvw = v * cone.luv - w
#         cone.is_feas = (cone.vluvw > 0)
#     else
#         cone.is_feas = false
#     end
#
#     cone.feas_updated = true
#     return cone.is_feas
# end
#
# function update_grad(cone::EpiPerExp)
#     @assert cone.is_feas
#     (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
#     vluvw = cone.vluvw
#
#     cone.g1a = -v / u / vluvw
#     cone.grad[1] = cone.g1a - inv(u)
#     cone.g2a = (1 - cone.luv) / vluvw
#     cone.grad[2] = cone.g2a - inv(v)
#     cone.grad[3] = inv(vluvw)
#
#     cone.grad_updated = true
#     return cone.grad
# end
#
# function update_hess(cone::EpiPerExp)
#     @assert cone.grad_updated
#     (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
#     H = cone.hess.data
#     vluvw = cone.vluvw
#     g1a = cone.g1a
#     g2a = cone.g2a
#
#     H[1, 3] = g1a / vluvw
#     H[2, 3] = g2a / vluvw
#     H[3, 3] = abs2(cone.grad[3])
#     H[1, 1] = abs2(g1a) - cone.grad[1] / u
#     H[1, 2] = -(v * cone.g2a + 1) / cone.vluvw / u
#     H[2, 2] = abs2(g2a) + (inv(vluvw) + inv(v)) / v
#
#     cone.hess_updated = true
#     return cone.hess
# end

# function update_inv_hess(cone::EpiPerExp)
#     @assert cone.is_feas
#     (u, v, w) = (cone.point[1], cone.point[2], cone.point[3])
#     Hi = cone.inv_hess.data
#     vluvw = cone.vluvw
#     vluv = vluvw + w
#     denom = vluvw + 2 * v
#     uvdenom = u * v / denom
#
#     Hi[1, 1] = u * (vluvw + v) / denom * u
#     Hi[2, 2] = v * (vluvw + v) / denom * v
#     Hi[3, 3] = 2 * (abs2(vluv - v) + vluv * (v - w)) + abs2(w) - v / denom * abs2(vluv - 2 * v)
#     Hi[1, 2] = uvdenom * v
#     Hi[1, 3] = uvdenom * (2 * vluv - w)
#     Hi[2, 3] = (abs2(vluv) + w * (v - vluv)) / denom * v
#
#     cone.inv_hess_updated = true
#     return cone.inv_hess
# end
#
# function update_inv_hess_prod(cone::EpiPerExp)
#     if !cone.inv_hess_updated
#         update_inv_hess(cone)
#     end
#     return
# end
#
# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerExp)
#     update_inv_hess_prod(cone)
#     mul!(prod, cone.inv_hess, arr)
#     return prod
# end
