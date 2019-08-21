#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

generalized power cone parametrized by alpha in R_+^n on unit simplex
(u in R^m, w in R_+^n) : norm_2(u) <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

barrier from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i((w_i)^(2 * alpha_i)) - norm_2(u)^2) - sum_i((1 - alpha_i)*log(w_i))
=#

mutable struct Power{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    alpha::Vector{T}
    m::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    logprodw::T
    prodw::T
    prodw_prodwu::T
    wiw::T
    prodwu::T
    alphawi::Vector{T}
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact # TODO prealloc
    cache_fact

    function Power{T}(alpha::Vector{T}, m::Int, is_dual::Bool) where {T <: Real}
        dim = length(alpha) + m
        @assert dim >= 3
        @assert all(ai > 0 for ai in alpha)
        tol = 1e3 * eps(T)
        @assert sum(alpha) ≈ 1 atol=tol rtol=tol
        cone = new{T}()
        cone.m = m
        cone.use_dual = is_dual
        cone.dim = dim
        cone.alpha = alpha
        return cone
    end
end

Power{T}(alpha::Vector{T}, m::Int) where {T <: Real} = Power{T}(alpha, m, false)

function setup_data(cone::Power{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.alphawi = zeros(T, dim - cone.m)
    cone.cache_fact = nothing
    return
end

get_nu(cone::Power) = length(cone.alpha) + 1

function set_initial_point(arr::AbstractVector, cone::Power)
    arr[1:cone.m] .= 0
    arr[(cone.m + 1):cone.dim] .= 1
    return arr
end

function update_feas(cone::Power)
    @assert !cone.feas_updated
    m = cone.m
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)
    if all(wi -> wi > 0, w)
        cone.logprodw = sum(cone.alpha[i] * log(w[i]) for i in eachindex(cone.alpha))
        cone.is_feas = (cone.logprodw > log(norm(u)))
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::Power)
    @assert cone.is_feas
    m = cone.m
    u = cone.point[1:cone.m]
    w = view(cone.point, (m + 1):cone.dim)
    @. cone.alphawi = 2 * cone.alpha / w
    # prod_i((w_i)^(2 * alpha_i))
    cone.prodw = exp(2 * cone.logprodw)
    # violation term
    cone.prodwu = cone.prodw - sum(abs2, u)
    # ratio of product and violation
    cone.prodw_prodwu = cone.prodw / cone.prodwu
    @. cone.grad[1:m] = u * 2 / cone.prodwu
    @. cone.grad[(m + 1):end] = -cone.alphawi * cone.prodw_prodwu - (1 - cone.alpha) / w
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Power)
    @assert cone.grad_updated
    m = cone.m
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)
    H = cone.hess.data
    alpha = cone.alpha
    alphawi = cone.alphawi
    prodw = cone.prodw
    prodwu = cone.prodwu
    prodw_prodwu = cone.prodw_prodwu

    # double derivative wrt u
    offset = 2 / prodwu
    scal = 2 * offset / prodwu
    @inbounds for j in 1:m
        @inbounds for i in 1:j
            H[i, j] = scal * u[i] * u[j]
        end
        H[j, j] += offset
    end

    @inbounds for j in eachindex(w)
        jm = j + m

        # derivative wrt u and w
        scal = -2 * alphawi[j] * prodw_prodwu / prodwu
        @inbounds for i in 1:m
            H[i, jm] = scal * u[i]
        end

        # double derivative wrt w
        awj_ratio = alphawi[j] * prodw_prodwu
        awj_ratio_ratio = awj_ratio * (prodw_prodwu - 1)
        @inbounds for i in 1:(j - 1)
            H[i + m, jm] = alphawi[i] * awj_ratio_ratio
        end
        H[jm, jm] = awj_ratio_ratio * alphawi[j] + (awj_ratio + (1 - alpha[j]) / w[j]) / w[j]
    end

    cone.hess_updated = true
    return cone.hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Power)
    @assert cone.grad_updated
    m = cone.m
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)
    alpha = cone.alpha
    prodwu = cone.prodwu
    alphawi = cone.alphawi
    prodw_prodwu = cone.prodw_prodwu
    offset = 2 / prodwu
    scal = 2 * offset / prodwu

    @views @inbounds for i in 1:size(arr, 2)
        # H[1:m, 1:m] * arr[1:m]
        @. prod[1:m, i] = u[1:m] * scal
        prod[1:m, i] .*= dot(u[1:m], arr[1:m, i])
        @. prod[1:m, i] += offset * arr[1:m, i]

        # H[(m + 1):dim, 1:m] * arr[(m + 1):dim]
        dotn = -2 * prodw_prodwu * dot(alphawi, arr[(m + 1):cone.dim, i]) / prodwu
        @. prod[1:m, i] += u[1:m] * dotn

        # H[(m + 1):dim, (m + 1):dim] * arr[(m + 1):dim]
        @. prod[(m + 1):cone.dim, i] = alphawi * prodw_prodwu * (prodw_prodwu - 1)
        prod[(m + 1):cone.dim, i] .*= dot(alphawi, arr[(m + 1):cone.dim, i])
        @. prod[(m + 1):cone.dim, i] += (prodw_prodwu * alphawi + (1 - alpha) / w) * arr[(m + 1):cone.dim, i] / w

        # H[1:m, (m + 1):dim] * arr[1:m]
        dotm = -2 * prodw_prodwu / prodwu * dot(u[1:m], arr[1:m, i])
        prod[(m + 1):cone.dim, i] .+= dotm * alphawi
    end

    return prod
end
