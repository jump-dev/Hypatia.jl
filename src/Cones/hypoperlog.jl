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
    hess_aux_updated::Bool
    inv_hess_aux_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    lwv::T
    z::T
    wivzi::Vector{T}
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

reset_data(cone::HypoPerLog) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated = cone.inv_hess_aux_updated = cone.hess_fact_updated = false)

function setup_extra_data(cone::HypoPerLog{T}) where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.wivzi = zeros(T, dim - 2)
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
    v = cone.point[2]
    @views w = cone.point[3:end]

    if v > eps(T) && all(>(eps(T)), w)
        u = cone.point[1]
        cone.lwv = sum(log, w) - length(w) * log(v)
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
    @views w = cone.dual_point[3:end]
    if all(>(eps(T)), w) && u < -eps(T)
        v = cone.dual_point[2]
        return (v - u * (sum(log, w) + length(w) * (1 - log(-u))) > eps(T))
    end
    return false
end

function update_grad(cone::HypoPerLog)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    d = length(w)
    z = cone.z
    g = cone.grad

    g[1] = inv(z)
    g[2] = (d - cone.lwv) / z - inv(v)
    zvzi = -(z + v) / z
    @inbounds @. @views g[3:end] = zvzi / w

    cone.grad_updated = true
    return cone.grad
end

# update first two rows of the Hessian
function update_hess_aux(cone::HypoPerLog)
    @assert !cone.hess_aux_updated
    @assert cone.grad_updated
    v = cone.point[2]
    @views w = cone.point[3:end]
    z = cone.z
    d = length(w)
    H = cone.hess.data
    dlzi = (d - cone.lwv) / z
    vzi = v / z
    wivzi = cone.wivzi
    @. wivzi = vzi / w

    @inbounds begin
        H[1, 1] = abs2(inv(z))
        H[1, 2] = H[2, 1] = dlzi / z
        @. @views H[1, 3:end] = wivzi / -z

        H[2, 2] = abs2(dlzi) + (d / z + inv(v)) / v
        H23const = -(v * dlzi + 1) / z
        @. @views H[2, 3:end] = H23const / w
    end

    cone.hess_aux_updated = true
    return
end

function update_hess(cone::HypoPerLog)
    if !cone.hess_aux_updated
        update_hess_aux(cone)
    end
    @views w = cone.point[3:end]
    H = cone.hess.data
    g = cone.grad
    wivzi = cone.wivzi

    @inbounds for j in eachindex(wivzi)
        j2 = 2 + j
        wivzij = wivzi[j]
        for i in 1:j
            H[2 + i, j2] = wivzi[i] * wivzij
        end
        H[j2, j2] -= g[j2] / w[j]
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPerLog)
    if !cone.hess_aux_updated
        update_hess_aux(cone)
    end
    H = cone.hess.data
    wivzi = cone.wivzi
    gww = cone.tempw
    @. @views gww = -cone.grad[3:end] ./ cone.point[3:end]

    @inbounds @views mul!(prod[1:2, :], H[1:2, :], arr)
    @inbounds for i in 1:size(arr, 2)
        @views arr_w = arr[3:end, i]
        dot_i = dot(wivzi, arr_w)
        @. @views prod[3:end, i] = dot_i * wivzi + gww * arr_w
    end
    @inbounds @views mul!(prod[3:end, :], H[1:2, 3:end]', arr[1:2, :], true, true)

    return prod
end

# update first two rows of the inverse Hessian
function update_inv_hess_aux(cone::HypoPerLog)
    @assert cone.feas_updated
    @assert !cone.inv_hess_aux_updated
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:end]
    Hi = cone.inv_hess.data
    d = length(w)
    z = cone.z
    zv = z + v
    zuz = 2 * z + u
    den = zv + d * v
    vden = v / den
    zuzvden = zuz * vden
    vvden = v * vden

    @inbounds begin
        Hi[1, 1] = abs2(z + u) + z * (den - v) - d * zuz * zuzvden
        Hi[1, 2] = Hi[2, 1] = vvden * (cone.lwv * (zv - d * v) + d * u)
        @. @views Hi[1, 3:end] = zuzvden * w

        Hi[2, 2] = vvden * zv
        @. @views Hi[2, 3:end] = vvden * w
    end

    cone.inv_hess_aux_updated = true
    return
end

function update_inv_hess(cone::HypoPerLog)
    if !cone.inv_hess_aux_updated
        update_inv_hess_aux(cone)
    end
    v = cone.point[2]
    @views w = cone.point[3:end]
    Hi = cone.inv_hess.data
    z = cone.z
    zv = z + v
    wzvi = cone.tempw
    @. wzvi = w / zv

    @inbounds for j in eachindex(w)
        j2 = 2 + j
        vwvdenj = Hi[2, j2]
        for i in 1:j
            Hi[2 + i, j2] = vwvdenj * wzvi[i]
        end
        Hi[j2, j2] += z * w[j] * wzvi[j]
    end

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPerLog)
    if !cone.inv_hess_aux_updated
        update_inv_hess_aux(cone)
    end
    v = cone.point[2]
    @views w = cone.point[3:end]
    Hi = cone.inv_hess.data
    z = cone.z
    zv = z + v
    wzvi = cone.tempw
    @. wzvi = w / zv
    @views vwvden = Hi[2, 3:end]

    @inbounds @views mul!(prod[1:2, :], Hi[1:2, :], arr)
    @inbounds for i in 1:size(arr, 2)
        @views arr_w = arr[3:end, i]
        dot_i = dot(vwvden, arr_w)
        @. @views prod[3:end, i] = (dot_i + z * arr_w * w) * wzvi
    end
    @inbounds @views mul!(prod[3:end, :], Hi[1:2, 3:end]', arr[1:2, :], true, true)

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

    d = length(w)
    z = cone.z
    lwvd = cone.lwv - d
    wdw = cone.tempw
    @. wdw = w_dir / w
    sumwdw = 2 * sum(wdw)
    sum2wdw = sum(abs2, wdw)
    vwdw = T(0.5) * v * sumwdw
    vz = v / z

    const1 = abs2(u_dir - lwvd * v_dir)
    const3 = vwdw * sumwdw / z + sum2wdw
    const4 = lwvd * v_dir
    const5 = 2 * (const4 - u_dir)
    const7 = 2 * lwvd - d
    const8 = 2 * ((u_dir - lwvd * v_dir) * vz + v_dir) / z
    const9 = v * sum2wdw + 2 * (const1 + vwdw * (const5 + vwdw)) / z
    const10 = ((v_dir * (-2 * u_dir + v_dir * const7) - 2 * const1 / z * v) / z + 2 * const8 * vwdw) / z - abs2(vz) * const3
    const11 = -abs2(vz) * sumwdw + const8
    const12 = -2 * (vz + 1)

    corr[1] = (const9 + v_dir * (v_dir * d / v - sumwdw)) / z / z

    corr[2] = ((-lwvd * const9 - d * v_dir * (const5 + const4) / v + sumwdw * (v_dir * const7 - u_dir)) / z + const3) / z - abs2(v_dir / v) * (d * inv(z) + 2 / v)

    @. w_corr = const11 .+ const12 * wdw
    w_corr .*= wdw
    w_corr .+= const10
    w_corr ./= w

    corr ./= -2

    return corr
end

# see analysis in https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
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
