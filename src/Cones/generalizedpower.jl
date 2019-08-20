#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

generalized power cone parametrized by alpha in R_+^n on unit simplex
(u in R^m, w in R_+^n) : norm_2(u) <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

barrier from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i((w_i)^(2 * alpha_i)) - norm_2(u)^2) - sum_i((1 - alpha_i)*log(w_i))
=#

mutable struct GeneralizedPower{T <: Real} <: Cone{T}
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

    wi2a::T
    wiw::T
    alphawi::Vector{T}
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact # TODO prealloc

    function GeneralizedPower{T}(alpha::Vector{T}, m::Int, is_dual::Bool) where {T <: Real}
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

GeneralizedPower{T}(alpha::Vector{T}, m::Int) where {T <: Real} = GeneralizedPower{T}(alpha, m, false)

function setup_data(cone::GeneralizedPower{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.alphawi = zeros(T, dim - 1)
    return
end

get_nu(cone::GeneralizedPower) = cone.dim

function set_initial_point(arr::AbstractVector, cone::GeneralizedPower)
    arr .= 1
    arr[1:cone.m] .= 0
    return arr
end

function update_feas(cone::GeneralizedPower)
    @assert !cone.feas_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    if all(wi -> wi > 0, w)
        cone.is_feas = (sum(cone.alpha[i] * log(w[i]) for i in eachindex(cone.alpha)) > log(sum(abs2, u)))
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::GeneralizedPower)
    @assert cone.is_feas
    m = cone.m
    u = cone.point[1:cone.m]
    w = view(cone.point, (m + 1):cone.dim)
    @. cone.alphawi = 2 * cone.alpha / w
    # prod_i((w_i)^(2 * alpha_i))
    cone.wi2a = exp(sum(2 * cone.alpha[i] * log(w[i]) for i in eachindex(cone.alpha)))
    # violation term
    wi2au = cone.wi2a - sum(abs2, u)
    @. cone.grad[1:m] = u * 2 / wi2au
    @. cone.grad[(m + 1):end] = -cone.alphawi * cone.wi2a / wi2au
    @. cone.grad[(m + 1):end] -= (1 - cone.alpha) / w
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::GeneralizedPower)
    @assert cone.grad_updated
    m = cone.m
    u = cone.point[1:m]
    w = view(cone.point, (m + 1):cone.dim)
    H = cone.hess.data
    alpha = cone.alpha
    alphawi = cone.alphawi

    wi2a = cone.wi2a
    wi2au = cone.wi2a - sum(abs2, u)
    # ratio of product and violation
    wi2a_wi2au = wi2a / wi2au

    H .= 0
    # double derivative wrt u
    scal = 4 / wi2au / wi2au
    offset = 2 / wi2au
    for i in 1:m
        H[i, i] = offset + scal * u[i]^2
    end
    for j in eachindex(w)
        jm = j + m
        # derivative wrt u and w
        scal = -4 * cone.alpha[j] * wi2a_wi2au / wi2au / w[j]
        for i in 1:m
            H[i, jm] = scal * u[i]
        end
        # double derivative wrt w
        awj_ratio = alphawi[j] * wi2a_wi2au
        awj_ratio_ratio = awj_ratio * (wi2a_wi2au - 1)
        for i in 1:(j - 1)
            im = i + m
            H[im, jm] = alphawi[i] * awj_ratio_ratio
        end
        H[jm, jm] = awj_ratio_ratio * alphawi[j] + (awj_ratio + (1 - alpha[j]) / w[j]) / w[j]
        # H[jm, jm] = -awj_ratio_ratio * cone.grad[jm] + (1 - cone.alpha[j]) / w[j] / w[j]
    end


    # reuse from gradient
    # term1 = cone.wi2a ./ w .* alpha * 2 / wi2au
    # for j in eachindex(w)
    #     j1 = j + 1
    #     H[1, j1] = -4 * cone.alpha[j] * u * wi2a_wi2au / wi2au / w[j]
    #     for i in 1:(j - 1)
    #         H[i + 1, j1] = 4 * alpha[i] * alpha[j] * wi2a_wi2au * (wi2a_wi2au - 1) / w[i] / w[j]
    #     end
    #     # H[j1, j1] = wiwaw * cone.grad[j1] + (1 - cone.alpha[j]) / w[j] / w[j]
    #     H[j1, j1] = term1[j]^2 - 2 * alpha[j] * (2 * alpha[j] - 1) * wi2a_wi2au / w[j]^2 + (1 - alpha[j]) / w[j]^2
    # end

    cone.hess_updated = true
    return cone.hess
end

# matrixified
# function update_hess(cone::GeneralizedPower)
#     @assert cone.grad_updated
#     u = cone.point[1]
#     w = view(cone.point, 2:cone.dim)
#     H = cone.hess.data
#     alpha = cone.alpha
#
#
#     wi2a = cone.wi2a
#     wi2au = cone.wi2a - abs2(u)
#     wi2a_wi2au = wi2a / wi2au
#
#     # reuse from gradient
#     term1 = cone.wi2a ./ w .* alpha * 2 / wi2au
#
#     H[1, 1] = 2 / wi2au + 4 * u^2 / wi2au^2
#     for j in eachindex(w)
#         j1 = j + 1
#         H[1, j1] = -4 * cone.alpha[j] * u * wi2a_wi2au / wi2au / w[j]
#     end
#
#     Hblock = view(H, 2:cone.dim, 2:cone.dim)
#     tmpvec = 2 * alpha ./ w
#     mul!(Hblock, tmpvec, tmpvec')
#     Hblock .*= wi2a_wi2au^2
#     Hblock .-= tmpvec * tmpvec' * wi2a_wi2au
#     for i in 1:(cone.dim - 1)
#         Hblock[i, i] += 2 * alpha[i] / w[i]^2 * wi2a_wi2au + (1 - alpha[i]) / w[i]^2
#     end
#
#     cone.hess_updated = true
#     return cone.hess
# end
