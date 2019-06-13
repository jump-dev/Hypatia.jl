#=
Copyright 2018, Chris Coey and contributors

(closure of) epigraph of perspective of sum of exponentials (n-dimensional exponential cone)
(u in R, v in R_+, w in R^n) : u >= v*sum(exp.(w/v))

barrier (guessed)
-log(v*log(u/v) - v*logsumexp(w/v)) - log(u) - log(v)
= -log(log(u) - log(v) - logsumexp(w/v)) - log(u) - 2*log(v)

TODO use the numerically safer way to evaluate LSE function
TODO compare alternative barrier -log(u - v*sum(wi -> exp(wi/v), w)) - log(u) - log(v)
=#

mutable struct EpiPerSumExp{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int

    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    F
    barfun::Function
    diffres

    function EpiPerSumExp{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiPerSumExp{T}(dim::Int) where {T <: HypReal} = EpiPerSumExp{T}(dim, false)

function setup_data(cone::EpiPerSumExp{T}) where {T <: HypReal}
    dim = cone.dim
    cone.g = zeros(T, dim)
    cone.H = zeros(T, dim, dim)
    cone.H2 = copy(cone.H)
    function barfun(point)
        u = point[1]
        v = point[2]
        w = view(point, 3:dim)
        # return -log(u - v*sum(wi -> exp(wi/v), w)) - log(u) - log(v)
        return -log(log(u) - log(v) - log(sum(wi -> exp(wi / v), w))) - log(u) - 2 * log(v) # TODO use the numerically safer way to evaluate LSE function
    end
    cone.barfun = barfun
    cone.diffres = DiffResults.HessianResult(cone.g)
    return
end

get_nu(cone::EpiPerSumExp) = 3

set_initial_point(arr::AbstractVector{T}, cone::EpiPerSumExp{T}) where {T <: HypReal} = (@. arr = -log(T(cone.dim - 2)); arr[1] = T(2); arr[2] = one(T); arr)

function check_in_cone(cone::EpiPerSumExp{T}) where {T <: HypReal}
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    if u <= zero(T) || v <= zero(T) || u <= v * sum(wi -> exp(wi / v), w) # TODO use the numerically safer way to evaluate LSE function
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.g .= DiffResults.gradient(cone.diffres)
    cone.H .= DiffResults.hessian(cone.diffres)

    return factorize_hess(cone)
end
