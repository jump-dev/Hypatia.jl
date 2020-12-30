#=
(closure of) epigraph of perspective of the quantum entropy function
(u in R, v in R_+, W in S_+^d) : u >= tr(W * log(W / v))

barrier -log(u - tr(W * log(W / v))) - log(v) - logdet(W) where log() is the matrix logarithm

TODO corrector
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
    dual_W::Matrix{T}
    fact_W
    z::T
    lwv::Matrix{T}
    tau::Matrix{T}
    sigma::T
    w_vals_log::Vector{T}
    diff_mat::Matrix{T}
    temp1::Vector{T}

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

# use_correction(::EpiPerTraceEntropyTri) = false

function setup_extra_data(cone::EpiPerTraceEntropyTri{T}) where T
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.lwv = zeros(T, cone.d, cone.d)
    cone.tau = zeros(T, cone.d, cone.d)
    cone.W = zeros(T, cone.d, cone.d)
    cone.dual_W = zeros(T, cone.d, cone.d)
    cone.diff_mat = zeros(T, cone.d, cone.d)
    cone.w_vals_log = zeros(T, cone.d)
    cone.temp1 = zeros(T, cone.dim - 2)
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
    @views W = Hermitian(svec_to_smat!(cone.W, cone.point[3:cone.dim], cone.rt2), :U)
    (w_vals, w_vecs) = cone.fact_W = eigen(W)

    if (v > eps(T)) && all(wi -> wi > eps(T), w_vals)
        @. cone.w_vals_log = log(w_vals)
        cone.lwv = w_vecs * Diagonal(cone.w_vals_log) * w_vecs'
        lv = log(v)
        for i in 1:cone.d
            cone.lwv[i, i] -= lv
        end
        cone.z = u - dot(W, Symmetric(cone.lwv, :U))
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
        @views W = Hermitian(svec_to_smat!(cone.dual_W, cone.dual_point[3:cone.dim], cone.rt2), :U)
        w_vals = eigvals!(W)
        @. w_vals /= -u
        @. w_vals -= 1
        return v - u * sum(exp, w_vals) > eps(T)
    end

    return false
end

function update_grad(cone::EpiPerTraceEntropyTri)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    W = Symmetric(cone.W, :U)
    # TODO use a different factorization?
    Wi = cone.Wi = inv(cone.fact_W)
    z = cone.z
    cone.sigma = tr(W) / v / z
    tau = cone.tau
    tau .= cone.lwv
    for i in 1:cone.d
        tau[i, i] += 1
    end
    @. tau /= z
    sigma = cone.sigma

    cone.grad[1] = -inv(z)
    cone.grad[2] = -sigma - inv(v)
    @views smat_to_svec!(cone.grad[3:end], tau - Wi, cone.rt2)

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
    sigma = cone.sigma
    H = cone.hess.data

    (w_vals, w_vecs) = cone.fact_W
    diff_mat!(diff_mat, w_vals, cone.w_vals_log)

    H[1, 1] = abs2(inv(z))
    H[1, 2] = sigma / z
    Huw = @views smat_to_svec!(H[1, 3:end], tau, cone.rt2)
    @. Huw /= -z
    H[2, 2] = abs2(sigma) - cone.grad[2] / v
    @views smat_to_svec!(H[2, 3:end], -sigma * tau - I / v / z, cone.rt2)
    @views symm_kron(H[3:end, 3:end], Wi, cone.rt2)
    tau_vec = smat_to_svec!(cone.temp1, tau, cone.rt2)
    @views mul!(H[3:end, 3:end], tau_vec, tau_vec', true, true)

    sdim = cone.dim - 2
    grad_logW = zeros(T, sdim, sdim)
    grad_logm!(grad_logW, w_vecs, diff_mat, cone.rt2)
    grad_logW ./= z
    @. @views H[3:end, 3:end] += grad_logW

    cone.hess_updated = true
    return cone.hess
end

using ForwardDiff
function logm(A)
    (vals, vecs) = eigen(Hermitian(A, :U))
    return vecs * Diagonal(log.(vals)) * vecs'
end
function correction(cone::EpiPerTraceEntropyTri{T}, primal_dir::AbstractVector{T}) where T
    @assert cone.hess_updated
    d = cone.d
    z = cone.z
    sigma = cone.sigma
    tau = cone.tau
    (w_vals, w_vecs) = cone.fact_W
    diff_mat = Symmetric(cone.diff_mat, :U)

    u = cone.point[1]
    v = cone.point[2]
    W = cone.W
    Wi = Symmetric(cone.Wi, :U)
    u_dir = primal_dir[1]
    v_dir = primal_dir[2]
    @views W_dir = Symmetric(svec_to_smat!(zeros(T, d, d), primal_dir[3:cone.dim], cone.rt2), :U)
    corr = cone.correction
    @views w_corr = corr[3:cone.dim]
    corr .= 0

    Tuuu = -2 / z / z / z
    Tuuv = -2 * sigma / z / z
    Tuuw = 2 * tau / z / z
    Tuvv = -2 * sigma ^ 2 / z - sigma / v / z
    Tuvw = 2 * sigma * tau / z + I / v / z / z
    Tvvv = -2 * sigma ^ 3 - 3 * sigma ^ 2 / v - 2 * sigma / v ^ 2 - 2 / v ^ 3
    Tvvw = tau * (2 * sigma ^ 2 + sigma / v) + I * (1 / v / v / z + 2 * sigma / v / z)

    temp1 = w_vecs' * W_dir * w_vecs
    temp2 = diff_mat .* temp1
    temp3 = dot(temp1', temp2)

    corr[1] = Tuuu * u_dir ^ 2 + 2 * Tuuv * u_dir * v_dir + 2 * dot(Tuuw, W_dir) * u_dir +
        Tuvv * v_dir ^ 2 + 2 * dot(Tuvw, W_dir) * v_dir
    corr[1] += -2 * dot(tau, W_dir)^2 / z - temp3 / z ^ 2
    corr[2] = Tuuv * u_dir ^ 2 + 2 * Tuvv * u_dir * v_dir + 2 * dot(Tuvw, W_dir) * u_dir +
        Tvvv * v_dir ^ 2 + 2 * dot(Tvvw, W_dir) * v_dir
    corr[2] -= 2 * tr(W_dir) * dot(tau, W_dir) / v / z + 2 * sigma * dot(tau, W_dir)^2 + sigma * temp3 / z

    X = w_vecs' * W_dir * w_vecs
    diff_tensor = zeros(T, d, d, d) # TODO reshape into a matrix
    diff_tensor!(diff_tensor, diff_mat, w_vals)
    diff_dot = [X[:, q]' * Diagonal(diff_tensor[:, p, q]) * X[:, p] for p in 1:d, q in 1:d]
    www_part = w_vecs * diff_dot * w_vecs'

    W_corr = Tuuw * u_dir ^ 2 + 2 * Tuvw * u_dir * v_dir + Tvvw * v_dir ^ 2 +
        2 * u_dir * (-2 * tau * dot(tau, W_dir) / z - w_vecs * temp2 * w_vecs' / z ^ 2) +
        2 * v_dir * (-(tau * tr(W_dir) + dot(W_dir, tau) * I) / v / z - 2 * sigma * tau * dot(tau, W_dir) - sigma / z * w_vecs * temp2 * w_vecs') +
        2 * tau * dot(W_dir, tau) ^ 2 +
        (tau * temp3 + dot(tau, W_dir) * w_vecs * temp2 * w_vecs' * 2) / z +
        2 * www_part / z -
        2 * Wi * W_dir * Wi * W_dir * Wi

    @views smat_to_svec!(w_corr, W_corr, cone.rt2)

    corr ./= -2

    return corr
end

# TODO reshape and pull into Cones.jl, also don't fill up entirely
function diff_tensor!(diff_tensor, diff_mat::AbstractMatrix{T}, w_vals) where T
    d = size(diff_mat, 1)
    for k in 1:d, j in 1:k, i in 1:j
        (vi, vj, vk) = (w_vals[i], w_vals[j], w_vals[k])
        if abs(vj - vk) < sqrt(eps(T))
            if abs(vi - vj) < sqrt(eps(T))
                diff_tensor[i, i, i] = -inv(vi) / vi / 2
            else
                diff_tensor[i, j, j] = diff_tensor[j, i, j] = diff_tensor[j, j, i] = -(inv(vj) - diff_mat[i, j]) / (vi - vj)
            end
        elseif abs(vi - vj) < sqrt(eps(T))
            diff_tensor[k, j, j] = diff_tensor[j, k, j] = diff_tensor[j, j, k] = (inv(vi) - diff_mat[k, i]) / (vi - vk)
        else
            diff_tensor[i, j, k] = diff_tensor[i, k, j] = diff_tensor[j, i, k] =
                diff_tensor[j, k, i] = diff_tensor[k, i, j] = diff_tensor[k, j, i] = (diff_mat[i, j] - diff_mat[k, i]) / (vj - vk)
        end
    end
    return diff_tensor
end
