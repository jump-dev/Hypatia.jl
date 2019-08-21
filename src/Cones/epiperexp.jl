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

mutable struct EpiPerExp{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
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

    barfun::Function
    diffres
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact
    hess_fact_cache

    function EpiPerExp{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        function barfun(point)
            u = point[1]
            v = point[2]
            w = view(point, 3:dim)
            # return -log(u - v*sum(wi -> exp(wi/v), w)) - log(u) - log(v)
            return -log(log(u) - log(v) - log(sum(wi -> exp(wi / v), w))) - log(u) - 2 * log(v) # TODO use the numerically safer way to evaluate LSE function
        end
        cone.barfun = barfun
        return cone
    end
end

EpiPerExp{T}(dim::Int) where {T <: Real} = EpiPerExp{T}(dim, false)

function setup_data(cone::EpiPerExp{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.diffres = DiffResults.HessianResult(cone.grad)
    cone.hess_fact_cache = nothing
    return
end

get_nu(cone::EpiPerExp) = 3

function set_initial_point(arr::AbstractVector{T}, cone::EpiPerExp{T}) where {T <: Real}
    @. arr = -log(T(cone.dim - 2))
    arr[1] = 2
    arr[2] = 1
    return arr
end

function update_feas(cone::EpiPerExp)
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    cone.is_feas = u > 0 && v > 0 && u > v * sum(wi -> exp(wi / v), w) # TODO use the numerically safer way to evaluate LSE function
    cone.feas_updated = true
    return cone.is_feas
end
