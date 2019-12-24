#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

generalized power cone parametrized by alpha in R_++^n in unit simplex interior
(u in R_++^m, w in R^n) : prod_i(u_i^alpha_i) => norm_2(w)
where sum_i(alpha_i) = 1, alpha_i > 0

barrier from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i((u_i)^(2 * alpha_i)) - norm_2(w)^2) - sum_i((1 - alpha_i)*log(u_i))
=#

mutable struct Power{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    alpha::Vector{T}
    n::Int
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

    logprodu::T
    produ::T
    produ_produw::T
    produw::T
    alphaui::Vector{T}
    tmpm::Vector{T}

    function Power{T}(
        alpha::Vector{T},
        n::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert n >= 1
        dim = length(alpha) + n
        @assert dim >= 3
        @assert all(ai > 0 for ai in alpha)
        tol = 1000 * eps(T)
        @assert sum(alpha) ≈ 1 atol=tol rtol=tol
        cone = new{T}()
        cone.n = n
        cone.use_dual = is_dual
        cone.dim = dim
        cone.alpha = alpha
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

Power{T}(alpha::Vector{T}, n::Int) where {T <: Real} = Power{T}(alpha, n, false)

# TODO only allocate the fields we use
function setup_data(cone::Power{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.alphaui = zeros(length(cone.alpha))
    cone.tmpm = zeros(length(cone.alpha))
    return
end

get_nu(cone::Power) = length(cone.alpha) + 1

function set_initial_point(arr::AbstractVector, cone::Power)
    m = length(cone.alpha)
    @. arr[1:m] = sqrt(1 + cone.alpha)
    arr[(m + 1):cone.dim] .= 0
    return arr
end

function update_feas(cone::Power)
    @assert !cone.feas_updated
    m = length(cone.alpha)
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)
    if all(ui -> ui > 0, u)
        cone.logprodu = sum(cone.alpha[i] * log(u[i]) for i in eachindex(cone.alpha))
        cone.is_feas = (cone.logprodu > log(norm(w)))
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::Power)
    @assert cone.is_feas
    m = length(cone.alpha)
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)
    @. cone.alphaui = 2 * cone.alpha / u
    # prod_i((u_i)^(2 * alpha_i))
    cone.produ = exp(2 * cone.logprodu)
    # violation term
    cone.produw = cone.produ - sum(abs2, w)
    # ratio of product and violation
    cone.produ_produw = cone.produ / cone.produw
    @. cone.grad[1:m] = -cone.alphaui * cone.produ_produw - (1 - cone.alpha) / u
    @. cone.grad[(m + 1):end] = w * 2 / cone.produw
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Power)
    @assert cone.grad_updated
    m = length(cone.alpha)
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)
    H = cone.hess.data
    alpha = cone.alpha
    alphaui = cone.alphaui
    produ = cone.produ
    produw = cone.produw
    produ_produw = cone.produ_produw

    # double derivative wrt u
    @inbounds for j in 1:m
        awj_ratio = alphaui[j] * produ_produw
        awj_ratio_ratio = awj_ratio * (produ_produw - 1)
        @inbounds for i in 1:j
            H[i, j] = alphaui[i] * awj_ratio_ratio
        end
        H[j, j] = awj_ratio_ratio * alphaui[j] + (awj_ratio + (1 - alpha[j]) / u[j]) / u[j]
    end

    offset = 2 / produw
    scal = 2 * offset / produw
    @. cone.tmpm = -2 * alphaui * produ_produw / produw
    for j in 1:cone.n
        jm = j + m

        # derivative wrt u and w
        @inbounds for i in 1:m
            H[i, jm] = cone.tmpm[i] * w[j]
        end

        # double derivative wrt w
        @inbounds for i in 1:j
            H[i + m, jm] = scal * w[i] * w[j]
        end
        H[jm, jm] += offset
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Power)
    @assert cone.grad_updated
    m = length(cone.alpha)
    dim = cone.dim
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):dim)
    alpha = cone.alpha
    produw = cone.produw
    tmpm = cone.tmpm
    alphaui = cone.alphaui
    produ_produw = cone.produ_produw
    @. tmpm = 2 * produ_produw * alphaui / produw

    @. @views prod[1:m, :] = alphaui * produ_produw * (produ_produw - 1)
    @. @views prod[(m + 1):dim, :] = 2 / produw
    @views @inbounds for i in 1:size(arr, 2)
        dotm = dot(alphaui, arr[1:m, i])
        dotn = dot(w, arr[(m + 1):dim, i])
        prod[1:m, i] .*= dotm
        prod[(m + 1):dim, i] .*= dotn
        @. prod[1:m, i] -= tmpm * dotn
        @. prod[(m + 1):dim, i] -= produ_produw * dotm
    end
    @. @views begin
        prod[1:m, :] += arr[1:m, :] * (produ_produw * alphaui + (1 - alpha) / u) / u
        prod[(m + 1):dim, :] *= w
        prod[(m + 1):dim, :] += arr[(m + 1):dim, :]
        prod[(m + 1):dim, :] *= 2
        prod[(m + 1):dim, :] /= produw
    end

    return prod
end
