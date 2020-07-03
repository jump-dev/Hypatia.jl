#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

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
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    produ::T
    produw::T
    produuw::T
    aui::Vector{T}
    auiproduuw::Vector{T}

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
function setup_data(cone::Power{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.aui = zeros(length(cone.alpha))
    cone.auiproduuw = zeros(length(cone.alpha))
    return
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
    for j in m .+ (1:cone.n)
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

# TODO improve / cache more, and benchmark to decide whether this improves speed/numerics
function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Power)
    @assert cone.grad_updated
    m = length(cone.alpha)
    dim = cone.dim
    @views u = cone.point[1:m]
    @views w = cone.point[(m + 1):end]
    alpha = cone.alpha
    produw = cone.produw
    aui = cone.aui
    produuw = cone.produuw
    # tmpm = cone.tmpm
    # @. tmpm = 2 * produuw * aui / produw # TODO cache
    tmpm = 2 * produuw * aui ./ produw

    @. @views prod[1:m, :] = aui * produuw * (produuw - 1) # TODO cache
    @. @views prod[(m + 1):dim, :] = 2 / produw
    @views @inbounds for i in 1:size(arr, 2)
        dotm = dot(aui, arr[1:m, i])
        dotn = dot(w, arr[(m + 1):dim, i])
        prod[1:m, i] .*= dotm
        prod[(m + 1):dim, i] .*= dotn
        @. prod[1:m, i] -= tmpm * dotn
        @. prod[(m + 1):dim, i] -= produuw * dotm
    end
    @. @views @inbounds begin
        prod[1:m, :] += arr[1:m, :] * (produuw * aui + (1 - alpha) / u) / u # TODO cache?
        prod[(m + 1):dim, :] *= w
        prod[(m + 1):dim, :] += arr[(m + 1):dim, :]
        prod[(m + 1):dim, :] *= 2
        prod[(m + 1):dim, :] /= produw
    end

    return prod
end

function correction(cone::Power, primal_dir::AbstractVector)
    @assert cone.hess_updated

    m = length(cone.alpha)
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)
    w_idxs = (m + 1):cone.dim
    alpha = cone.alpha
    T = eltype(cone.point)

    produ = cone.produ # = exp(2 * sum(cone.alpha[i] * log(u[i]) for i in eachindex(cone.alpha)))
    produw = cone.produw # = cone.produ - sum(abs2, w)
    produuw = cone.produuw # = cone.produ / cone.produw
    produuw_tw = produuw * (produuw - 1)
    aui = cone.aui # @. cone.aui = 2 * cone.alpha / u

    corr = cone.correction
    corr .= 0
    dim = cone.dim
    u_corr = view(corr, 1:m)
    w_corr = view(corr, w_idxs)
    u_dir = view(primal_dir, 1:m)
    w_dir = view(primal_dir, w_idxs)

    wwdir = dot(w, w_dir)
    auiudir = dot(aui, u_dir)

    u_corr .+=
        # uuu
        produuw * (1 - produuw) * aui .*
        ((2 * produuw - 1) * abs2(auiudir) + dot(aui ./ u, abs2.(u_dir)) .+ 2 * auiudir * u_dir ./ u) +
        -2 * ((1 .- alpha) ./ u + produuw * aui) ./ u ./ u .* abs2.(u_dir) +
        # uuw
        2 * produuw / produw * aui .*
        (wwdir * (
        (2 * produuw - 1) * 2 * auiudir .+
        2 * u_dir ./ u .+
        # uww
        -4 / produw * wwdir
        ) .-
        sum(abs2.(w_dir))
        )

    # w[wk] / produw / produw
    w_corr .+=
        # uuw
        2 * produuw / produw * w .* ((2 * produuw - 1) * abs2(auiudir) + dot(aui ./ u, abs2.(u_dir))) +
        # uww
        -4 * produuw / produw * auiudir * (4  / produw * wwdir * w + w_dir) +
        # www
        4 / produw / produw * (
        4 * abs2(wwdir) / produw * w +
        2 * wwdir .* w_dir + sum(abs2.(w_dir)) .* w
        )
    corr ./= -2

    return cone.correction
end
