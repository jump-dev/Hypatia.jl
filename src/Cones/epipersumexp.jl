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

mutable struct EpiPerSumExp <: Cone
    usedual::Bool
    dim::Int
    primals::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function EpiPerSumExp(dim::Int, isdual::Bool)
        cone = new()
        cone.usedual = isdual
        cone.dim = dim
        cone.g = Vector{Float64}(undef, dim)
        cone.H = Matrix{Float64}(undef, dim, dim)
        cone.H2 = similar(cone.H)
        function barfun(primals)
            u = primals[1]
            v = primals[2]
            w = view(primals, 3:dim)
            # return -log(u - v*sum(wi -> exp(wi/v), w)) - log(u) - log(v)
            return -log(log(u) - log(v) - log(sum(wi -> exp(wi / v), w))) - log(u) - 2.0 * log(v) # TODO use the numerically safer way to evaluate LSE function
        end
        cone.barfun = barfun
        cone.diffres = DiffResults.HessianResult(cone.g)
        return cone
    end
end

EpiPerSumExp(dim::Int) = EpiPerSumExp(dim, false)

get_nu(cone::EpiPerSumExp) = 3 # TODO does this increase with dim > 3?

set_initial_point(arr::AbstractVector{Float64}, cone::EpiPerSumExp) = (@. arr = -log(cone.dim - 2); arr[1] = 2.0; arr[2] = 1.0; arr)

function check_in_cone(cone::EpiPerSumExp)
    u = cone.primals[1]
    v = cone.primals[2]
    w = view(cone.primals, 3:cone.dim)
    if u <= 0.0 || v <= 0.0 || u <= v * sum(wi -> exp(wi / v), w) # TODO use the numerically safer way to evaluate LSE function
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.primals)
    cone.g .= DiffResults.gradient(cone.diffres)
    cone.H .= DiffResults.hessian(cone.diffres)

    return factorize_hess(cone)
end
