#=
generalized power cone parametrized by alpha in R_++^n in unit simplex interior
(u in R_++^m, w in R^n) : prod_i(u_i^alpha_i) => norm_2(w)
where sum_i(alpha_i) = 1, alpha_i > 0

barrier from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i((u_i)^(2 * alpha_i)) - norm_2(w)^2) - sum_i((1 - alpha_i)*log(u_i))
=#

mutable struct Power{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    alpha::Vector{T}
    n::Int

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

    produ::T
    produw::T
    produuw::T
    aui::Vector{T}
    auiproduuw::Vector{T}
    tmpm::Vector{T}

    function Power{T}(
        alpha::Vector{T},
        n::Int;
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert n >= 1
        dim = length(alpha) + n
        @assert dim >= 3
        @assert all(ai > 0 for ai in alpha)
        @assert sum(alpha) ≈ 1
        cone = new{T}()
        cone.n = n
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.alpha = alpha
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

dimension(cone::Power) = length(cone.alpha) + cone.n

# TODO only allocate the fields we use
function setup_extra_data(cone::Power{T}) where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    m = length(cone.alpha)
    cone.aui = zeros(T, m)
    cone.auiproduuw = zeros(T, m)
    cone.tmpm = zeros(T, m)
    return cone
end

get_nu(cone::Power) = length(cone.alpha) + 1

function set_initial_point(arr::AbstractVector, cone::Power)
    m = length(cone.alpha)
    @. arr[1:m] = sqrt(1 + cone.alpha)
    arr[(m + 1):cone.dim] .= 0
    return arr
end

function update_feas(cone::Power{T}) where {T <: Real}
    @assert !cone.feas_updated
    m = length(cone.alpha)
    @views u = cone.point[1:m]

    if all(>(eps(T)), u)
        @inbounds cone.produ = exp(2 * sum(cone.alpha[i] * log(u[i]) for i in eachindex(cone.alpha)))
        @views cone.produw = cone.produ - sum(abs2, cone.point[(m + 1):end])
        cone.is_feas = (cone.produw > eps(T))
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function is_dual_feas(cone::Power{T}) where {T <: Real}
    alpha = cone.alpha
    m = length(cone.alpha)
    @views u = cone.dual_point[1:m]

    if all(>(eps(T)), u)
        @inbounds p = exp(2 * sum(alpha[i] * log(u[i] / alpha[i]) for i in eachindex(alpha)))
        @views w = cone.dual_point[(m + 1):end]
        return (p - sum(abs2, w) > eps(T))
    end

    return false
end

function update_grad(cone::Power)
    @assert cone.is_feas
    m = length(cone.alpha)
    @views u = cone.point[1:m]
    @views w = cone.point[(m + 1):end]

    @. cone.aui = 2 * cone.alpha / u
    cone.produuw = cone.produ / cone.produw
    @. cone.auiproduuw = -cone.aui * cone.produuw
    @. cone.grad[1:m] = cone.auiproduuw - (1 - cone.alpha) / u
    produwi2 = 2 / cone.produw
    @. cone.grad[(m + 1):end] = produwi2 * w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Power)
    @assert cone.grad_updated
    m = length(cone.alpha)
    @views u = cone.point[1:m]
    @views w = cone.point[(m + 1):end]
    aui = cone.aui
    auiproduuw = cone.auiproduuw
    g = cone.grad
    H = cone.hess.data

    produuwm1 = 1 - cone.produuw
    @inbounds for j in 1:m
        auiproduuwm1 = auiproduuw[j] * produuwm1
        @inbounds for i in 1:j
            H[i, j] = aui[i] * auiproduuwm1
        end
        H[j, j] -= g[j] / u[j]
    end

    offset = 2 / cone.produw
    @inbounds for j in m .+ (1:cone.n)
        gj = g[j]
        @inbounds for i in 1:m
            H[i, j] = auiproduuw[i] * gj
        end
        @inbounds for i in (m + 1):j
            H[i, j] = g[i] * gj
        end
        H[j, j] += offset
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Power)
    @assert cone.grad_updated
    m = length(cone.alpha)
    @views u = cone.point[1:m]
    @views w = cone.point[(m + 1):end]
    aui = cone.aui
    produuw = cone.produuw
    w_idxs = (m + 1):cone.dim
    produwi2 = 2 / cone.produw
    const1 = 2 * produuw - 1
    tmpm = cone.tmpm
    @. tmpm = (1 + const1 * cone.alpha) / u / u

    @views @inbounds for i in 1:size(arr, 2)
        arr_u = arr[1:m, i]
        arr_w = arr[w_idxs, i]
        dot1 = -produuw * dot(aui, arr_u)
        dot2 = dot1 + produwi2 * dot(w, arr_w)
        dot3 = dot1 - produuw * dot2
        dot4 = produwi2 * dot2
        @. prod[1:m, i] = dot3 * aui + tmpm * arr_u
        @. prod[w_idxs, i] = dot4 * w + produwi2 * arr_w
    end

    return prod
end

function correction(cone::Power, primal_dir::AbstractVector)
    m = length(cone.alpha)
    @views u = cone.point[1:m]
    @views w = cone.point[(m + 1):end]
    corr = cone.correction
    @views u_corr = corr[1:m]
    @views w_corr = corr[(m + 1):end]
    @views u_dir = primal_dir[1:m]
    @views w_dir = primal_dir[(m + 1):end]
    alpha = cone.alpha
    produw = cone.produw
    produuw = cone.produuw

    wwd = 2 * dot(w, w_dir)
    udu = cone.tmpm
    @. udu = u_dir / u
    audu = dot(alpha, udu)
    const8 = 2 * produuw - 1
    const1 = 2 * const8 * abs2(audu) + sum(ai * udui * udui for (ai, udui) in zip(alpha, udu))
    const15 = wwd / produw
    const10 = sum(abs2, w_dir) + wwd * const15

    const11 = -2 * produuw * (1 - produuw)
    const12 = -2 * produuw / produw
    const13 = const11 * const1 + const12 * (2 * wwd * const8 * audu - const10)
    const14 = const11 * 2 * audu + const12 * wwd
    @. u_corr = const14 .+ const8 * udu
    u_corr .*= alpha
    u_corr .+= udu
    u_corr .*= udu
    @. u_corr += const13 * alpha
    u_corr ./= u

    const2 = -2 * const12 * audu
    const6 = 2 * const2 * const15 + const12 * const1 - 2 / produw * const10 / produw
    const7 = const2 - 2 / produw * wwd / produw
    @. w_corr = const7 * w_dir + const6 * w

    return cone.correction
end
