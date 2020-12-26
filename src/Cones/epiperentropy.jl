#=
(closure of) epigraph of perspective of the entropy function on R^d
(u in R, v in R_+, w in R_+^d) : u >= sum_i w_i*log(w_i / v)

barrier -log(u - sum_i w_i*log(w_i / v)) - log(v) - sum_i log(w_i)

TODO
simplify corrector
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
    v = cone.dual_point[2]
    @views w = cone.dual_point[3:cone.dim]
    if u > eps(T)
        return v - u * sum(exp(-wi ./ u - 1) for wi in w) > eps(T)
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

function update_hess(cone::EpiPerEntropy{T}) where T
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
    tau = -cone.tau
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

    i2z = inv(2 * z)
    @. wdw = w_dir / w
    vdv = v_dir / v
    const0 = (u_dir + sum(w) * vdv) / z + dot(tau, w_dir)
    const1 = abs2(const0) + sum(w[i] * abs2(vdv) + w_dir[i] * (wdw[i] - 2 * vdv) for i in eachindex(w)) / (2 * z)
    corr[1] = const1 / z

    # v
    # corr[2] = const1
    # corr[2] += (const0 + vdv) * vdv - i2z * sum(wdw) * sum(w_dir)
    # corr[2] *= sum(w)
    # # corr[2] += (z * vdv - sum(w_dir)) * vdv + (-const0 + i2z * sum(w_dir)) * sum(w_dir)
    # corr[2] += (z * vdv - sum(w_dir)) * vdv + (-const0) * sum(w_dir)
    # corr[2] /= v
    # corr[2] /= z

    sigma = sum(w) / v / z
    Tuuv = -2 * sigma / z^2
    Tuvv = -2 * sigma^2 / z - sigma / v / z
    Tuvw = 2 * sigma * -tau / z .+ inv(v) / abs2(z)
    Tvvv = -2 * sigma^3 - 3 * (sigma^2 / v) - 2 * sigma / abs2(v) - 2 / v ^ 3
    Tvvw = 2 * sigma^2 * -tau + (sigma * -tau / v) .+ 2 * sigma / v / z .+ inv(abs2(v)) / z
    Tvww = Diagonal(-sigma ./ w ./ z) + -2 * tau * tau' * sigma .+ tau / z / v .+ tau' / z / v

    corr[2] = Tuuv * u_dir^2 + 2 * (Tuvv * u_dir * v_dir + dot(Tuvw, w_dir) * u_dir + dot(Tvvw, w_dir) * v_dir) + Tvvv * v_dir^2 + w_dir' * Tvww * w_dir
    corr[2] /= -2

    # w
    @. w_corr = const1 * tau
    @. w_corr += ((const0 - w * vdv / z) / z + (inv(w) + i2z) * wdw) * wdw
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
