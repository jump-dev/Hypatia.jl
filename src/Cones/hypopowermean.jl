#=
hypograph of power mean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

barrier from "Constructing self-concordant barriers for convex cones" by Yu. Nesterov
-log(prod_i(w_i^alpha_i) - u) - sum_i(log(w_i))
=#

mutable struct HypoPowerMean{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    dim::Int
    alpha::Vector{T}

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

    wprod::T
    z::T
    tempw::Vector{T}

    function HypoPowerMean{T}(
        alpha::Vector{T};
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        dim = length(alpha) + 1
        @assert dim >= 2
        @assert all(ai > 0 for ai in alpha)
        @assert sum(alpha) ≈ 1
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.dim = dim
        cone.alpha = alpha
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data(cone::HypoPowerMean{T}) where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.tempw = zeros(T, dim - 1)
    return cone
end

get_nu(cone::HypoPowerMean) = cone.dim

function set_initial_point(arr::AbstractVector{T}, cone::HypoPowerMean{T}) where T
    # get closed form central ray if all powers are equal, else use fitting
    if all(isequal(inv(T(cone.dim - 1))), cone.alpha)
        n = cone.dim - 1
        c = sqrt(T(5 * n ^ 2 + 2 * n + 1))
        arr[1] = -sqrt((-c + 3 * n + 1) / T(2 * n + 2))
        @views arr[2:end] .= (c - n + 1) / sqrt(T(n + 1) * (-2 * c + 6 * n + 2))
    else
        (arr[1], w) = get_central_ray_hypopowermean(cone.alpha)
        @views arr[2:end] = w
    end
    return arr
end

function update_feas(cone::HypoPowerMean{T}) where T
    @assert !cone.feas_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    alpha = cone.alpha

    if all(>(eps(T)), w)
        @inbounds cone.wprod = exp(sum(alpha[i] * log(w[i]) for i in eachindex(alpha)))
        cone.z = cone.wprod - u
        cone.is_feas = (cone.z > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::HypoPowerMean{T}) where T
    u = cone.dual_point[1]
    @views w = cone.dual_point[2:end]
    alpha = cone.alpha

    @inbounds if u < -eps(T) && all(>(eps(T)), w)
        return (exp(sum(alpha[i] * log(w[i] / alpha[i]) for i in eachindex(alpha))) + u > eps(T))
    end

    return false
end

function update_grad(cone::HypoPowerMean)
    @assert cone.is_feas
    u = cone.point[1]
    @views w = cone.point[2:end]

    cone.grad[1] = inv(cone.z)
    wwprodu = -cone.wprod / cone.z
    @. @views cone.grad[2:end] = (wwprodu * cone.alpha - 1) / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPowerMean)
    @assert cone.grad_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    alpha = cone.alpha
    z = cone.z
    aw = alpha ./ w # TODO
    wwprodu = cone.wprod / z
    wwprodum1 = wwprodu - 1
    H = cone.hess.data

    H[1, 1] = abs2(cone.grad[1])
    @inbounds for j in eachindex(w)
        j1 = j + 1
        wj = w[j]
        aj = alpha[j]
        awwwprodu = wwprodu * aw[j]
        H[1, j1] = -awwwprodu / z
        awwwprodum1 = awwwprodu * wwprodum1
        @inbounds for i in 1:(j - 1)
            H[i + 1, j1] = awwwprodum1 * aw[i]
        end
        H[j1, j1] = (awwwprodu * (1 + aj * wwprodum1) + inv(wj)) / wj
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPowerMean)
    @assert cone.grad_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    alpha = cone.alpha
    z = cone.z
    wwprodu = cone.wprod / z
    wwprodum1 = wwprodu - 1
    aw = alpha ./ w # TODO
    awwwprodu = wwprodu * aw # TODO

    @inbounds @views for j in 1:size(arr, 2)
        arr_u = arr[1, j]
        arr_w = arr[2:end, j]
        auz = arr_u / z
        prod[1, j] = (auz - dot(awwwprodu, arr_w)) / z
        dot1 = -auz + wwprodum1 * dot(arr_w, aw)
        @. prod[2:end, j] = dot1 * awwwprodu + (awwwprodu * arr_w + arr_w / w) / w
    end

    return prod
end

function correction(cone::HypoPowerMean, primal_dir::AbstractVector)
    @assert cone.grad_updated
    u = cone.point[1]
    @views w = cone.point[2:end]
    u_dir = primal_dir[1]
    @views w_dir = primal_dir[2:end]
    corr = cone.correction
    @views wcorr = corr[2:end]
    pi = cone.wprod # TODO rename
    z = cone.z
    alpha = cone.alpha
    wdw = cone.tempw

    piz = pi / z
    @. wdw = w_dir / w
    udz = u_dir / z
    const6 = -2 * udz * piz
    awdw = dot(alpha, wdw)
    const1 = awdw * piz * (2 * piz - 1)
    awdw2 = sum(alpha[i] * abs2(wdw[i]) for i in eachindex(alpha))
    corr[1] = (abs2(udz) + const6 * awdw + (const1 * awdw + piz * awdw2) / 2) / -z
    const2 = piz * (1 - piz)
    const3 = (const6 * u_dir / z + const2 * awdw2 - u / z * const1 * awdw) / -2 - udz * const1
    const4 = const2 * awdw + udz * piz
    const5 = const2 + piz * u / z

    @. wcorr = piz + const5 * alpha
    wcorr .*= alpha
    wcorr .+= 1
    wcorr .*= wdw
    @. wcorr -= const4 * alpha
    wcorr .*= wdw
    @. wcorr += const3 * alpha
    wcorr ./= w

    return corr
end

# see analysis in https://github.com/lkapelevich/HypatiaSupplements.jl/tree/master/centralpoints
function get_central_ray_hypopowermean(alpha::Vector{<:Real})
    wdim = length(alpha)
    # predict each w_i given alpha_i and n
    w = zeros(wdim)
    if wdim == 1
        w .= 1.306563
    elseif wdim == 2
        @. w = 1.0049885 + 0.2986276 * alpha
    elseif wdim <= 5
        @. w = 1.0040142949 - 0.0004885108 * wdim + 0.3016645951 * alpha
    elseif wdim <= 20
        @. w = 1.001168 - 4.547017e-05 * wdim + 3.032880e-01 * alpha
    elseif wdim <= 100
        @. w = 1.000069 - 5.469926e-07 * wdim + 3.074084e-01 * alpha
    else
        @. w = 1 + 3.086535e-01 * alpha
    end
    # get u in closed form from w
    p = exp(sum(alpha[i] * log(w[i]) for i in eachindex(alpha)))
    u = sum(p .- alpha .* p ./ (abs2.(w) .- 1)) / wdim
    return [u, w]
end
