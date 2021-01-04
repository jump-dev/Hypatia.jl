#=
(closure of) epigraph of perspective of the entropy function on R^d
(u in R, v in R_+, w in R_+^d) : u >= sum_i w_i*log(w_i / v)

barrier -log(u - sum_i w_i*log(w_i / v)) - log(v) - sum_i log(w_i)
=#

mutable struct EpiPerEntropy{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    w_dim::Int

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

    z::T
    lwv::Vector{T}
    tau::Vector{T}
    wi::Vector{T}
    temp1::Vector{T}
    sigma::T

    function EpiPerEntropy{T}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.w_dim = dim - 2
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

use_heuristic_neighborhood(cone::EpiPerEntropy) = false

function setup_extra_data(cone::EpiPerEntropy{T}) where T
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.lwv = zeros(T, cone.w_dim)
    cone.tau = zeros(T, cone.w_dim)
    cone.wi = zeros(T, cone.w_dim)
    cone.temp1 = zeros(T, cone.w_dim)
    return cone
end

get_nu(cone::EpiPerEntropy) = cone.dim

function set_initial_point(arr::AbstractVector{T}, cone::EpiPerEntropy{T}) where T
    (u, w) = get_central_ray_epiperentropy(cone.w_dim)
    arr[1] = u
    arr[2] = sqrt(T(cone.w_dim) * w * u + one(T))
    @views arr[3:cone.dim] .= w
    return arr
end

function update_feas(cone::EpiPerEntropy{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:cone.dim]

    if (v > eps(T)) && all(wi -> wi > eps(T), w)
        @. cone.lwv = log(w / v)
        cone.z = u - dot(w, cone.lwv)
        cone.is_feas = (cone.z > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiPerEntropy{T}) where T
    u = cone.dual_point[1]
    if u > eps(T)
        v = cone.dual_point[2]
        @views w = cone.dual_point[3:cone.dim]
        @. cone.temp1 = -w / u - 1
        return v - u * sum(exp, cone.temp1) > eps(T)
    end

    return false
end

function update_grad(cone::EpiPerEntropy)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:cone.dim]
    z = cone.z
    cone.sigma = sum(w) / v / z
    @. cone.tau = (cone.lwv + 1) / z
    @. cone.wi = inv(w)

    cone.grad[1] = -inv(z)
    cone.grad[2] = -cone.sigma - inv(v)
    @. cone.grad[3:end] = cone.tau - cone.wi

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiPerEntropy)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:cone.dim]
    z = cone.z
    tau = cone.tau
    sigma = cone.sigma
    H = cone.hess.data

    H[1, 1] = -cone.grad[1] / z
    H[1, 2] = sigma / z
    @. @views H[1, 3:end] = -tau / z
    H[2, 2] = abs2(sigma) - cone.grad[2] / v
    @. @views H[2, 3:end] = -sigma * tau - inv(v) / z
    @views Hww = H[3:end, 3:end]
    mul!(Hww, tau, tau')
    @inbounds for i in 1:cone.w_dim
        Hww[i, i] += (inv(z) + cone.wi[i]) / w[i]
    end

    cone.hess_updated = true
    return cone.hess
end

function correction(cone::EpiPerEntropy{T}, primal_dir::AbstractVector{T}) where T
    @assert cone.grad_updated
    tau = cone.tau
    z = cone.z
    u = cone.point[1]
    v = cone.point[2]
    @views w = cone.point[3:cone.dim]
    u_dir = primal_dir[1]
    v_dir = primal_dir[2]
    @views w_dir = primal_dir[3:cone.dim]
    corr = cone.correction
    @views w_corr = corr[3:cone.dim]
    wdw = similar(w)

    @. wdw = w_dir / w
    vdv = v_dir / v
    sw = sum(w)
    swd = sum(w_dir)
    const0 = (u_dir + sw * vdv) / z - dot(tau, w_dir)
    const1 = abs2(const0) + sw * abs2(vdv) / (2 * z) - swd * vdv / z + sum(w_dir[i] * wdw[i] for i in eachindex(w)) / (2 * z)
    corr[1] = const1 / z

    # v
    corr[2] = const1
    corr[2] += vdv * (const0 + vdv)
    corr[2] *= sw
    corr[2] += vdv * (z * vdv - swd) + (-const0) * swd
    corr[2] /= v
    corr[2] /= z

    # w
    @. w_corr = -const1 * tau
    @. w_corr += ((const0 - w * vdv / z) / z + (inv(w) + inv(2 * z)) * wdw) * wdw
    @. w_corr += (-const0 + w_dir / z - vdv / 2) / z * vdv

    return corr
end

# see analysis in https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
function get_central_ray_epiperentropy(w_dim::Int)
    if w_dim <= 10
        return central_rays_epiperentropy[w_dim, :]
    end
    # use nonlinear fit for higher dimensions
    invdim = inv(w_dim)
    u = 1.852439 / w_dim + 0.000355
    w = -0.413194 / w_dim + 0.999736
    return [u, w]
end

const central_rays_epiperentropy = [
    0.827838375	0.805102008;
    0.645834031	0.85849039;
    0.508867461	0.890884144;
    0.412577808	0.911745085;
    0.344114874	0.926086794;
    0.293923494	0.936489183;
    0.255930316	0.944357104;
    0.226334824	0.950507083;
    0.202707993	0.955442166;
    0.183449604	0.959487861;
    ]
