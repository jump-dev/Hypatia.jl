#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

generalized power cone parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : |u| <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

barrier from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i((w_i)^(2 * alpha_i)) - norm_2(u)) - sum_i((1 - alpha_i)*log(w_i))
=#

mutable struct GeneralizedPower{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    alpha::Vector{T}
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

    function GeneralizedPower{T}(alpha::Vector{T}, is_dual::Bool) where {T <: Real}
        dim = length(alpha) + 1
        @assert dim >= 3
        @assert all(ai > 0 for ai in alpha)
        tol = 1e3 * eps(T)
        @assert sum(alpha) ≈ 1 atol=tol rtol=tol
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.alpha = alpha
        return cone
    end
end

GeneralizedPower{T}(alpha::Vector{T}) where {T <: Real} = GeneralizedPower{T}(alpha, false)

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
    arr[1] = 0
    return arr
end

function update_feas(cone::GeneralizedPower)
    @assert !cone.feas_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    if all(wi -> wi > 0, w)
        cone.is_feas = (sum(cone.alpha[i] * log(w[i]) for i in eachindex(cone.alpha)) > log(abs(u)))
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::GeneralizedPower)
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    @. cone.alphawi = 2 * cone.alpha / w
    # prod_i((w_i)^(2 * alpha_i)) - norm_2(u))
    cone.wi2a = exp(sum(2 * cone.alpha[i] * log(w[i]) for i in eachindex(cone.alpha)))
    # violation term
    wi2au = cone.wi2a - abs2(u)
    cone.grad[1] = 2 * u / wi2au
    @. cone.grad[2:end] = -cone.alphawi * cone.wi2a / wi2au
    @. cone.grad[2:end] -= (1 - cone.alpha) / w
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::GeneralizedPower)
    @assert cone.grad_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    H = cone.hess.data
    alpha = cone.alpha
    alphawi = cone.alphawi

    wi2a = cone.wi2a
    wi2au = cone.wi2a - abs2(u)
    # ratio of product and violation
    wi2a_wi2au = wi2a / wi2au

    H[1, 1] = 2 / wi2au + 4 * u^2 / wi2au^2
    for j in eachindex(w)
        j1 = j + 1
        H[1, j1] = -4 * cone.alpha[j] * u * wi2a_wi2au / wi2au / w[j]
        awj_ratio = alphawi[j] * wi2a_wi2au
        awj_ratio_ratio = awj_ratio * (wi2a_wi2au - 1)
        for i in 1:(j - 1)
            H[i + 1, j1] = alphawi[i] * awj_ratio_ratio
        end
        H[j1, j1] = awj_ratio_ratio * alphawi[j] + (awj_ratio + (1 - alpha[j]) / w[j]) / w[j]
        # H[j1, j1] = -awj_ratio_ratio * cone.grad[j1] + (1 - cone.alpha[j]) / w[j] / w[j]
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
