#=
(closure of) hypograph of perspective of sum of logarithms
(u in R, v in R_+, w in R_+^d) : u <= v*sum(log.(w/v))

barrier modified from "Primal-Dual Interior-Point Methods for Domain-Driven Formulations" by Karimi & Tuncel, 2019
-log(sum_i v*log(w_i/v) - u) - sum_i log(w_i) - log(v)
=#

mutable struct HypoPerLog{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    correction::Vector{T}
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
    hess_fact_cache

    lwv::T
    z::T
    lvwniz::T
    tempw::Vector{T}

    function HypoPerLog{T}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

use_heuristic_neighborhood(cone::HypoPerLog) = false

function setup_extra_data(cone::HypoPerLog{T}) where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.tempw = zeros(T, dim - 2)
    return cone
end

get_nu(cone::HypoPerLog) = cone.dim

function set_initial_point(arr::AbstractVector, cone::HypoPerLog)
    (arr[1], arr[2], w) = get_central_ray_hypoperlog(cone.dim - 2)
    @views arr[3:end] .= w
    return arr
end

function update_feas(cone::HypoPerLog{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]

    if v > eps(T) && all(>(eps(T)), w)
        cone.lwv = sum(log(wi / v) for wi in w)
        cone.z = v * cone.lwv - u
        cone.is_feas = (cone.z > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoPerLog{T}) where T
    u = cone.dual_point[1]
    v = cone.dual_point[2]
    @views w = cone.dual_point[3:cone.dim]

    if all(>(eps(T)), w) && u < -eps(T)
        return (v - u * (length(w) + sum(log(-wi / u) for wi in w)) > eps(T))
    end

    return false
end

function update_grad(cone::HypoPerLog)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
    g = cone.grad

    g[1] = inv(cone.z)
    cone.lvwniz = (d - cone.lwv) / cone.z
    g[2] = cone.lvwniz - inv(v)
    gden = -1 - v / cone.z
    @. @views g[3:end] = gden / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLog)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
    tempw = cone.tempw
    lvwniz = cone.lvwniz
    viz = v / cone.z
    @. tempw = viz / w
    g = cone.grad
    H = cone.hess.data

    @inbounds begin
        H[1, 1] = abs2(g[1])
        H[1, 2] = lvwniz / cone.z
        @. @views H[1, 3:end] = -tempw / cone.z

        H[2, 2] = abs2(lvwniz) + (d * g[1] + inv(v)) / v
        hden = (-v * lvwniz - 1) / cone.z
        @. @views H[2, 3:end] = hden / w
    end

    @inbounds for j in 1:d
        j2 = 2 + j
        tempwj = tempw[j]
        @inbounds for i in 1:j
            H[2 + i, j2] = tempw[i] * tempwj
        end
        H[j2, j2] -= g[j2] / w[j]
    end

    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess(cone::HypoPerLog)
    @assert cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
    Hi = cone.inv_hess.data
    lwv = cone.lwv
    z = cone.z
    zv = z + v
    zu = z + u
    den = zv + d * v
    vden = v / den
    tempw = cone.tempw

    @inbounds begin
        Hi[1, 1] = abs2(zu) + z * (den - v) - (d * v * abs2(zu + z)) / den
        Hi[1, 2] = v * (lwv * (zv - d * v) + d * u) * vden
        Hi3const = (zu + z) * vden
        @. @views Hi[1, 3:end] = Hi3const * w

        @. tempw = w * v
        Hi[2, 2] = v * zv * vden
        @. @views Hi[2, 3:end] = vden * tempw
    end

    @. tempw = w / zv
    @inbounds for j in 1:d
        j2 = 2 + j
        vwvdenj = Hi[2, j2]
        for i in 1:(j - 1)
            Hi[2 + i, j2] = vwvdenj * tempw[i]
        end
        Hi[j2, j2] = (vwvdenj + w[j] * z) * tempw[j]
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPerLog)
    if !cone.inv_hess_updated
        update_inv_hess(cone)
    end
    mul!(prod, cone.inv_hess, arr)
    return prod
end

function correction(cone::HypoPerLog{T}, primal_dir::AbstractVector{T}) where {T <: Real}
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    u_dir = primal_dir[1]
    v_dir = primal_dir[2]
    @views w_dir = primal_dir[3:end]
    corr = cone.correction
    @views w_corr = corr[3:end]

    w_dim = length(w)
    z = v * sum(log(wi / v) for wi in w) - u # TODO cache?
    dzdv = -w_dim * log(v) - w_dim + sum(log, w)
    wdw = cone.tempw
    @. wdw = w_dir / w
    sumwdw = 2 * sum(wdw)
    sum2wdw = sum(abs2, wdw)
    vwdw = T(0.5) * v * sumwdw
    vd2 = abs2(v_dir)
    vz = v / z
    const1 = abs2(u_dir - dzdv * v_dir)
    const3 = vwdw * sumwdw / z + sum2wdw
    const4 = v_dir * dzdv
    const5 = 2 * (const4 - u_dir)
    const7 = 2 * dzdv - w_dim
    const8 = 2 * ((u_dir - dzdv * v_dir) * vz + v_dir) / z
    const9 = v * sum2wdw + 2 * (const1 + vwdw * (const5 + vwdw)) / z
    const10 = ((-2 * u_dir * v_dir + vd2 * const7 - 2 * const1 / z * v) / z + 2 * const8 * vwdw) / z - abs2(vz) * const3
    const11 = -abs2(vz) * sumwdw + const8
    const12 = -2 * (vz + 1)

    corr[1] = (const9 - sumwdw * v_dir + w_dim * vd2 / v) / z / z

    corr[2] = ((-dzdv * const9 - w_dim * v_dir * (const5 + const4) / v + sumwdw * (v_dir * const7 - u_dir)) / z + const3) / z - abs2(v_dir / v) * (w_dim * inv(z) + 2 / v)

    @. w_corr = const11 .+ const12 * wdw
    w_corr .*= wdw
    w_corr .+= const10
    w_corr ./= w

    corr ./= -2

    return corr
end

# see analysis in https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
function get_central_ray_hypoperlog(w_dim::Int)
    if w_dim <= 10
        # lookup points where x = f'(x)
        return central_rays_hypoperlog[w_dim, :]
    end
    # use nonlinear fit for higher dimensions
    x = inv(w_dim)
    if w_dim <= 70
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
