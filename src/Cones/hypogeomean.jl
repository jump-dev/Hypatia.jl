#=
Copyright 2018, Chris Coey and contributors

hypograph of generalized geomean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

dual barrier (modified by reflecting around u = 0 and using dual cone definition) from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i((w_i/alpha_i)^alpha_i) + u) - sum_i((1 - alpha_i)*log(w_i/alpha_i)) - log(-u)

TODO try to make barrier evaluation more efficient
=#

mutable struct HypoGeomean{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    alpha::Vector{<:Real}

    ialpha::Vector{<:Real}
    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    F
    barfun::Function
    diffres

    function HypoGeomean{T}(alpha::Vector{<:Real}, is_dual::Bool) where {T <: HypReal}
        dim = length(alpha) + 1
        @assert dim >= 3
        @assert all(ai >= 0.0 for ai in alpha)
        tol = 1e3 * eps(T)
        @assert sum(alpha) ≈ 1 atol=tol rtol=tol
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.alpha = alpha
        return cone
    end
end

HypoGeomean{T}(alpha::Vector{<:Real}) where {T <: HypReal} = HypoGeomean{T}(alpha, false)

function setup_data(cone::HypoGeomean{T}) where {T <: HypReal}
    dim = cone.dim
    alpha = cone.alpha
    ialpha = inv.(alpha)
    cone.ialpha = ialpha
    cone.g = zeros(T, dim)
    cone.H = zeros(T, dim, dim)
    cone.H2 = copy(cone.H)
    function barfun(point)
        u = point[1]
        w = view(point, 2:dim)
        return -log(prod((w[i] * ialpha[i])^alpha[i] for i in eachindex(alpha)) + u) - sum((1.0 - alpha[i]) * log(w[i] * ialpha[i]) for i in eachindex(alpha)) - log(-u)
    end
    cone.barfun = barfun
    cone.diffres = DiffResults.HessianResult(cone.g)
    return
end

get_nu(cone::HypoGeomean) = cone.dim

set_initial_point(arr::AbstractVector{T}, cone::HypoGeomean{T}) where {T <: HypReal} = (@. arr = one(T); arr[1] = -prod(cone.ialpha[i]^cone.alpha[i] for i in eachindex(cone.alpha)) / cone.dim; arr)

function check_in_cone(cone::HypoGeomean{T}) where {T <: HypReal}
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    if u >= zero(T) || any(wi <= zero(T) for wi in w)
        return false
    end
    if sum(cone.alpha[i] * log(w[i] * cone.ialpha[i]) for i in eachindex(cone.alpha)) <= log(-u)
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.g .= DiffResults.gradient(cone.diffres)
    cone.H .= DiffResults.hessian(cone.diffres)

    return factorize_hess(cone)
end
