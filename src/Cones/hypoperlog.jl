#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of sum of logarithms
(u in R, v in R_+, w in R_+^d) : u <= v*sum(log.(w/v))

barrier modified from "Primal-Dual Interior-Point Methods for Domain-Driven Formulations" by Karimi & Tuncel, 2019
-log(sum_i v*log(w_i/v) - u) - sum_i log(w_i) - d*log(v)
=#

mutable struct HypoPerLog{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::Vector{T}
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

    lwv::T
    vlwvu::T
    lvwnivlwvu::T
    vwivlwvu::Vector{T}

    function HypoPerLog{T}(
        dim::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoPerLog{T}(dim::Int) where {T <: Real} = HypoPerLog{T}(dim, false)

# TODO only allocate the fields we use
function setup_data(cone::HypoPerLog{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.vwivlwvu = zeros(T, dim - 2)
    return
end

get_nu(cone::HypoPerLog) = cone.dim

function set_initial_point(arr::AbstractVector, cone::HypoPerLog)
    (arr[1], arr[2], w) = get_central_ray_hypoperlog(cone.dim - 2)
    arr[3:end] .= w
    return arr
end

function update_feas(cone::HypoPerLog)
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)

    if v > 0 && all(>(zero(u)), w)
        cone.lwv = sum(log(wi / v) for wi in w)
        cone.vlwvu = v * cone.lwv - u
        cone.is_feas = (cone.vlwvu > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoPerLog)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    g = cone.grad

    g[1] = inv(cone.vlwvu)
    cone.lvwnivlwvu = (length(cone.vwivlwvu) - cone.lwv) / cone.vlwvu
    g[2] = cone.lvwnivlwvu - inv(v)
    gden = -1 - inv(cone.lwv - u / v)
    @. g[3:end] = gden / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLog)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    vwivlwvu = cone.vwivlwvu
    lvwnivlwvu = cone.lvwnivlwvu
    g = cone.grad
    H = cone.hess.data

    vivlwvu = v / cone.vlwvu
    @. vwivlwvu = vivlwvu / w
    H[1, 1] = abs2(g[1])
    H[1, 2] = lvwnivlwvu / cone.vlwvu
    @. H[1, 3:end] = -vwivlwvu / cone.vlwvu
    H[2, 2] = abs2(lvwnivlwvu) + (length(vwivlwvu) / cone.vlwvu + inv(v)) / v
    hden = (-v * lvwnivlwvu - 1) / cone.vlwvu
    @. H[2, 3:end] = hden / w
    @inbounds for j in eachindex(vwivlwvu)
        j2 = 2 + j
        @inbounds for i in 1:j
            H[2 + i, j2] = vwivlwvu[i] * vwivlwvu[j]
        end
        H[j2, j2] -= g[j2] / w[j]
    end

    cone.hess_updated = true
    return cone.hess
end

# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
function get_central_ray_hypoperlog(w_dim::Int)
    if w_dim <= 10
        # lookup points where x = f'(x)
        return central_rays_hypoperlog[w_dim, :]
    end
    # use nonlinear fit for higher dimensions
    if w_dim <= 70
        u = -2.647364 / w_dim - 0.008411
        v = 0.424679 / w_dim + 0.553392
        w = 0.760415 / w_dim + 1.001795
    else
        u = -3.016339 / w_dim - 0.000078
        v = 0.394332 / w_dim + 0.553963
        w = 0.838584 / w_dim + 1.000016
    end
    return [u, v, w]
end

const central_rays_hypoperlog = [
    -0.827838399  0.805102005  1.290927713;
    -0.689609381  0.724604185  1.224619879;
    -0.584372734  0.681280549  1.182421998;
    -0.503500819  0.654485416  1.153054181;
    -0.440285901  0.636444221  1.131466932;
    -0.389979933  0.623569273  1.114979598;
    -0.349256801  0.613977662  1.102014462;
    -0.315769104  0.60658984   1.091577909;
    -0.287837755  0.600745276  1.083013006;
    -0.264242795  0.596018958  1.075868819;
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
