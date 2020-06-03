#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

hypograph of generalized geomean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

barrier from "Constructing self-concordant barriers for convex cones" by Yu. Nesterov
-log(prod_i(w_i^alpha_i) - u) - sum_i(log(w_i))
=#

mutable struct HypoGeomean{T <: Real} <: Cone{T}
    use_dual_barrier::Bool
    use_heuristic_neighborhood::Bool
    max_neighborhood::T
    dim::Int
    alpha::Vector{T}
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
    old_hess
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache
    nbhd_tmp::Vector{T}
    nbhd_tmp2::Vector{T}

    wprod::T
    z::T

    correction::Vector{T}

    function HypoGeomean{T}(
        alpha::Vector{T};
        use_dual::Bool = false,
        use_heuristic_neighborhood::Bool = default_use_heuristic_neighborhood(),
        max_neighborhood::Real = default_max_neighborhood(),
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        dim = length(alpha) + 1
        @assert dim >= 2
        @assert all(ai > 0 for ai in alpha)
        @assert sum(alpha) ≈ 1
        cone = new{T}()
        cone.use_dual_barrier = use_dual
        cone.use_heuristic_neighborhood = use_heuristic_neighborhood
        cone.max_neighborhood = max_neighborhood
        cone.dim = dim
        cone.alpha = alpha
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_data(cone::HypoGeomean{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.old_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.nbhd_tmp = zeros(T, dim)
    cone.nbhd_tmp2 = zeros(T, dim)
    cone.correction = zeros(T, dim)
    return
end

get_nu(cone::HypoGeomean) = cone.dim

use_correction(cone::HypoGeomean) = true

use_scaling(cone::HypoGeomean) = false

rescale_point(cone::HypoGeomean{T}, s::T) where {T} = (cone.point .*= s)

function set_initial_point(arr::AbstractVector{T}, cone::HypoGeomean{T}) where {T}
    # get closed form central ray if all powers are equal, else use fitting
    if all(isequal(inv(T(cone.dim - 1))), cone.alpha)
        n = cone.dim - 1
        c = sqrt(T(5 * n ^ 2 + 2 * n + 1))
        arr[1] = -sqrt((-c + 3 * n + 1) / T(2 * n + 2))
        arr[2:end] .= (c - n + 1) / sqrt(T(n + 1) * (-2 * c + 6 * n + 2))
    else
        (arr[1], w) = get_central_ray_hypogeomean(cone.alpha)
        arr[2:end] = w
    end
    return arr
end

function update_feas(cone::HypoGeomean)
    @assert !cone.feas_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)

    if all(>(zero(u)), w)
        cone.wprod = exp(sum(cone.alpha[i] * log(w[i]) for i in eachindex(cone.alpha)))
        cone.z = cone.wprod - u
        cone.is_feas = (u < 0) || (cone.z > 0)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_dual_feas(cone::HypoGeomean{T}) where {T <: Real}
    u = cone.dual_point[1]
    w = view(cone.dual_point, 2:cone.dim)
    alpha = cone.alpha
    if u < 0
        return u < exp(sum(alpha[i] * log(w[i] / alpha[i]) for i in eachindex(alpha)))
    else
        return false
    end
end

function update_grad(cone::HypoGeomean)
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)

    cone.grad[1] = inv(cone.z)
    wwprodu = -cone.wprod / cone.z
    @. cone.grad[2:end] = (wwprodu * cone.alpha - 1) / w

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoGeomean)
    @assert cone.grad_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    alpha = cone.alpha
    z = cone.z
    wwprodu = cone.wprod / z
    wwprodum1 = wwprodu - 1
    H = cone.hess.data

    H[1, 1] = cone.grad[1] / z
    @inbounds for j in eachindex(w)
        j1 = j + 1
        wj = w[j]
        aj = alpha[j]
        awwwprodu = wwprodu / wj * aj
        H[1, j1] = -awwwprodu / z
        awwwprodum1 = awwwprodu * wwprodum1
        @inbounds for i in 1:(j - 1)
            H[i + 1, j1] = awwwprodum1 * alpha[i] / w[i]
        end
        H[j1, j1] = (awwwprodu * (1 + aj * wwprodum1) + inv(wj)) / wj
    end

    copyto!(cone.old_hess.data, H)

    cone.hess_updated = true
    return cone.hess
end

# TODO move into new cone with equal alphas
function update_inv_hess(cone::HypoGeomean{T}) where {T}
    @assert all(isequal(inv(T(cone.dim - 1))), cone.alpha)
    @assert !cone.inv_hess_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    n = cone.dim - 1
    wprod = cone.wprod
    H = cone.inv_hess.data
    denom = n * (n + 1) * wprod - abs2(n) * u
    H[1, 1] = (n + 1) * abs2(wprod) / n + abs2(u) - 2 * wprod * u
    H[1, 2:end] = wprod .* w / n
    H[2:end, 2:end] = wprod * w * w'
    H[2:end, 2:end] .+= Diagonal(abs2.(w) .* cone.z .* abs2(n))
    H[2:end, 2:end] /= denom

    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::HypoGeomean{T}) where {T}
    @assert all(isequal(inv(T(cone.dim - 1))), cone.alpha)
    dim = cone.dim
    u = cone.point[1]
    w = view(cone.point, 2:dim)
    n = dim - 1
    wprod = cone.wprod

    @inbounds for j in 1:size(prod, 2)
        @views pa = dot(w, arr[2:dim, j]) * wprod
        @. prod[2:dim, j] = pa * w
        prod[1, j] = pa / n
    end
    @. @views prod[1, :] += ((n + 1) * abs2(wprod) / n + abs2(u) - 2 * wprod * u) * arr[1, :]
    @. @views prod[2:dim, :] += (abs2.(w) .* cone.z .* abs2(n)) * arr[2:dim, :]
    @. @views prod[2:dim, :] /= (n * (n + 1) * wprod - abs2(n) * u)
    @views mul!(prod[2:dim, :], w, arr[1, :]', wprod / n, true)
    return prod
end

using ForwardDiff

function correction(
    cone::HypoGeomean{T},
    primal_dir::AbstractVector{T},
    dual_dir::AbstractVector{T},
    ) where {T <: Real}
    if !cone.hess_updated
        update_hess(cone)
    end
    dim = cone.dim
    u = cone.point[1]
    w = view(cone.point, 2:dim)
    wprod = cone.wprod
    z = cone.z
    alpha = cone.alpha
    pi = prod(w[j] ^ alpha[j] for j in eachindex(w))
    wwprodu = wprod / z
    tau = alpha ./ w ./ z
    alpha = cone.alpha
    Hinv_z = similar(dual_dir)
    inv_hess_prod!(Hinv_z, dual_dir, cone)

    third = zeros(T, dim, dim, dim)
    # Tuuu
    third[1, 1, 1] = 2 / z ^ 3
    # Tuuw
    for i in eachindex(w)
        i1 = i + 1
        third[1, 1, i1] = third[1, i1, 1] = third[i1, 1, 1] = -2 * tau[i] * pi / abs2(z)
    end
    # Tuww
    for i in eachindex(w), j in eachindex(w)
        (i1, j1) = (i + 1, j + 1)
        t1 = pi * tau[i] * tau[j] * (2 * pi / z - 1)
        if i == j
            third[i1, i1, 1] = third[i1, 1, i1] = third[1, i1, i1] = t1 + tau[i] * pi / w[i] / z
        else
            third[1, i1, j1] = third[1, j1, i1] = third[i1, 1, j1] = third[j1, 1, i1] =
                third[i1, j1, 1] = third[j1, i1, 1] = t1
        end
    end
    # Twww
    sigma = alpha ./ w
    for i in eachindex(w), j in eachindex(w), k in eachindex(w)
        (i1, j1, k1) = (i + 1, j + 1, k + 1)
        t1 = u * pi / abs2(z) * sigma[i] * sigma[j] * sigma[k] * (1 - 2 * pi / z)
        if i == j
            t2 = pi * sigma[i] * sigma[k] / w[i] / z * (1 - pi / z)
            if j == k
                third[i1, i1, i1] = t1 + t2 - 2 * pi * tau[i] / w[i] * (u * tau[i] + inv(w[i])) - 2 / w[i] ^ 3
            else
                third[i1, i1, k1] = third[i1, k1, i1] = third[k1, i1, i1] = t1 + t2
            end
        elseif i != k && j != k
            third[i1, j1, k1] = third[i1, k1, j1] = third[j1, i1, k1] = third[j1, k1, i1] =
                third[k1, i1, j1] = third[k1, j1, i1] = t1
        end
    end
    third_order = reshape(third, dim ^ 2, dim)

    # function barrier(s)
    #     (u, w) = (s[1], s[2:end])
    #     return -log(prod(w[j] ^ alpha[j] for j in eachindex(w)) - u) - sum(log(wi) for wi in w)
    # end
    # FD_3deriv = ForwardDiff.jacobian(x -> ForwardDiff.hessian(barrier, x), cone.point)
    # @show (third_order - FD_3deriv)

    cone.correction = reshape(third_order * primal_dir, dim, dim) * Hinv_z
    cone.correction ./= -2

    return cone.correction
end


# see analysis in https://github.com/lkapelevich/HypatiaBenchmarks.jl/tree/master/centralpoints
function get_central_ray_hypogeomean(alpha::Vector{<:Real})
    wdim = length(alpha)
    # predict each w_i given alpha_i and n
    w = zeros(wdim)
    if wdim == 1
        w .= 1.306563
    elseif wdim == 2
        @. w = 1.005320 + 0.298108 * alpha
    elseif wdim <= 5
        @. w = 1.0049327 - 0.0006020 * wdim + 0.2998672 * alpha
    elseif wdim <= 20
        @. w = 1.001146 - 4.463046e-05 * wdim + 3.034014e-01 * alpha
    elseif wdim <= 100
        @. w = 1.000066 - 5.202270e-07 * wdim + 3.074873e-01 * alpha
    else
        @. w = 1 + 3.086695e-01 * alpha
    end
    # get u in closed form from w
    p = exp(sum(alpha[i] * log(w[i]) for i in eachindex(alpha)))
    u = sum(p .- alpha .* p ./ (abs2.(w) .- 1)) / wdim
    return [u, w]
end
