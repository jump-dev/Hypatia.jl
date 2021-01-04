#=
(closure of) epigraph of perspective of the quantum entropy function
(u in R, v in R_+, W in S_+^d) : u >= tr(W * log(W / v))

barrier -log(u - tr(W * log(W / v))) - log(v) - logdet(W) where log() is the matrix logarithm
=#

mutable struct EpiPerTraceEntropyTri{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    d::Int
    rt2::T

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

    W::Matrix{T}
    Wi::Matrix{T}
    fact_W
    trW::T
    z::T
    lwv::Matrix{T}
    tau::Matrix{T}
    sigma::T
    W_vals_log::Vector{T}
    diff_mat::Matrix{T}
    temp1::Vector{T}
    mat::Matrix{T}
    mat2::Matrix{T}
    matsdim1::Matrix{T}
    matsdim2::Matrix{T}
    grad_logW::Matrix{T}

    function EpiPerTraceEntropyTri{T}(
        dim::Int;
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.d = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
        cone.rt2 = sqrt(T(2))
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

use_heuristic_neighborhood(::EpiPerTraceEntropyTri) = false

function setup_extra_data(cone::EpiPerTraceEntropyTri{T}) where T
    dim = cone.dim
    sdim = cone.dim - 2
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.lwv = zeros(T, cone.d, cone.d)
    cone.tau = zeros(T, cone.d, cone.d)
    cone.W = zeros(T, cone.d, cone.d)
    cone.diff_mat = zeros(T, cone.d, cone.d)
    cone.W_vals_log = zeros(T, cone.d)
    cone.temp1 = zeros(T, sdim)
    cone.mat = zeros(T, cone.d, cone.d)
    cone.mat2 = zeros(T, cone.d, cone.d)
    cone.matsdim1 = zeros(T, sdim, sdim)
    cone.matsdim2 = zeros(T, sdim, sdim)
    cone.grad_logW = zeros(T, sdim, sdim)
    return cone
end

get_nu(cone::EpiPerTraceEntropyTri) = cone.d + 2

function set_initial_point(arr::AbstractVector{T}, cone::EpiPerTraceEntropyTri{T}) where T
    d = cone.d
    (u, w) = get_central_ray_epiperentropy(d)
    arr[1] = u
    arr[2] = sqrt(T(d) * w * u + one(T))
    for i in 1:d
        arr[2 + sum(1:i)] = w
    end
    return arr
end

function update_feas(cone::EpiPerTraceEntropyTri{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    @views W = svec_to_smat!(cone.W, cone.point[3:cone.dim], cone.rt2)
    copyto!(cone.mat2, W)
    (W_vals, W_vecs) = cone.fact_W = eigen!(Hermitian(cone.mat2, :U))

    if (v > eps(T)) && all(wi -> wi > eps(T), W_vals)
        @. cone.W_vals_log = log(W_vals)
        mul!(cone.mat, W_vecs, Diagonal(cone.W_vals_log))
        mul!(cone.lwv, cone.mat, W_vecs')
        lv = log(v)
        @inbounds for i in 1:cone.d
            cone.lwv[i, i] -= lv
        end
        cone.z = u - dot(Hermitian(W, :U), Hermitian(cone.lwv, :U))
        cone.is_feas = (cone.z > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::EpiPerTraceEntropyTri{T}) where T
    u = cone.dual_point[1]
    if u > eps(T)
        v = cone.dual_point[2]
        @views W = Hermitian(svec_to_smat!(cone.mat, cone.dual_point[3:cone.dim], cone.rt2), :U)
        W_vals = eigvals!(W)
        @. W_vals /= -u
        @. W_vals -= 1
        return v - u * sum(exp, W_vals) > eps(T)
    end

    return false
end

function update_grad(cone::EpiPerTraceEntropyTri)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    W = Hermitian(cone.W, :U)
    Wi = cone.Wi = inv(cone.fact_W)
    z = cone.z
    cone.trW = tr(W)
    cone.sigma = cone.trW / v / z
    tau = cone.tau
    copyto!(tau, cone.lwv)
    @inbounds for i in 1:cone.d
        tau[i, i] += 1
    end
    @. tau /= z
    sigma = cone.sigma

    cone.grad[1] = -inv(z)
    cone.grad[2] = -sigma - inv(v)
    @. cone.mat = tau - Wi
    @views smat_to_svec!(cone.grad[3:end], cone.mat, cone.rt2)

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiPerTraceEntropyTri{T}) where T
    @assert cone.grad_updated
    d = cone.d
    u = cone.point[1]
    v = cone.point[2]
    W = Hermitian(cone.W, :U)
    Wi = Hermitian(cone.Wi, :U)
    diff_mat = cone.diff_mat
    z = cone.z
    tau = cone.tau
    trW = cone.trW
    sigma = cone.sigma
    sigtau_vz = cone.mat
    H = cone.hess.data

    (W_vals, W_vecs) = cone.fact_W
    diff_mat!(diff_mat, W_vals, cone.W_vals_log)

    H[1, 1] = abs2(inv(z))
    H[1, 2] = sigma / z
    Huw = @views smat_to_svec!(H[1, 3:end], tau, cone.rt2)
    @. Huw /= -z
    H[2, 2] = abs2(sigma) - cone.grad[2] / v
    @. sigtau_vz = -trW * tau
    @inbounds for i in 1:d
        sigtau_vz[i, i] -= 1
    end
    @. sigtau_vz /= v * z
    @views smat_to_svec!(H[2, 3:end], sigtau_vz, cone.rt2)
    @views symm_kron(H[3:end, 3:end], Wi, cone.rt2)
    tau_vec = smat_to_svec!(cone.temp1, tau, cone.rt2)
    @views mul!(H[3:end, 3:end], tau_vec, tau_vec', true, true)

    grad_logW = cone.grad_logW
    grad_logm!(grad_logW, W_vecs, cone.matsdim1, cone.matsdim2, cone.temp1, diff_mat, cone.rt2)
    @. grad_logW /= z
    @. @views H[3:end, 3:end] += grad_logW

    cone.hess_updated = true
    return cone.hess
end

function correction(cone::EpiPerTraceEntropyTri{T}, primal_dir::AbstractVector{T}) where T
    @assert cone.hess_updated
    d = cone.d
    z = cone.z
    tau = cone.tau
    (W_vals, W_vecs) = cone.fact_W
    diff_mat = Symmetric(cone.diff_mat, :U)
    trW = cone.trW

    u = cone.point[1]
    v = cone.point[2]
    W = cone.W
    Wi = Symmetric(cone.Wi, :U)
    u_dir = primal_dir[1]
    v_dir = primal_dir[2]
    @views W_dir = Symmetric(svec_to_smat!(zeros(T, d, d), primal_dir[3:cone.dim], cone.rt2), :U)
    corr = cone.correction
    @views w_corr = corr[3:cone.dim]

    temp1 = W_vecs' * W_dir * W_vecs
    temp2 = diff_mat .* temp1
    temp3 = dot(temp1', temp2)
    vdv = v_dir / v
    trWd = tr(W_dir)
    const0 = (u_dir + trW * vdv) / z - dot(tau, W_dir)
    const1 = abs2(const0) + trW * abs2(vdv) / (2 * z) - trWd * vdv / z + temp3 / (2 * z)
    const2 = vdv * (vdv / 2 + const0) / z

    # u
    corr[1] = const1 / z

    # v
    corr[2] = const1
    corr[2] += vdv * (const0 + vdv)
    corr[2] *= trW
    corr[2] += vdv * (z * vdv - trWd) + (-const0) * trWd
    corr[2] /= v
    corr[2] /= z

    # W
    diff_tensor = zeros(T, d, d, d)
    diff_tensor!(diff_tensor, diff_mat, W_vals)
    diff_dot = [temp1[:, q]' * Diagonal(diff_tensor[:, p, q]) * temp1[:, p] for p in 1:d, q in 1:d]
    www_part_1 = W_vecs * diff_dot * W_vecs'
    wdw = W_vecs * temp2 * W_vecs'
    sqrt_vals = sqrt.(W_vals)
    www_half = W_vecs / Diagonal(W_vals) * temp1 / Diagonal(sqrt_vals)
    www_part_2 = www_half * www_half'
    W_corr = -tau * const1 + wdw * const0 / z - www_part_1 / z + www_part_2
    for i in 1:d
        W_corr[i, i] -= const2
    end
    @views smat_to_svec!(w_corr, W_corr, cone.rt2)

    return corr
end
