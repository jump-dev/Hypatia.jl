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
    hess_fact # TODO prealloc

    function EpiPerSumExp{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiPerSumExp{T}(dim::Int) where {T <: HypReal} = EpiPerSumExp{T}(dim, false)

reset_data(cone::EpiPerSumExp) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_prod_updated = false)

function setup_data(cone::EpiPerSumExp{T}) where {T <: HypReal}
    reset_data(cone)
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    function barfun(point)
        u = point[1]
        v = point[2]
        w = view(point, 3:dim)
        # return -log(u - v*sum(wi -> exp(wi/v), w)) - log(u) - log(v)
        return -log(log(u) - log(v) - log(sum(wi -> exp(wi / v), w))) - log(u) - 2 * log(v) # TODO use the numerically safer way to evaluate LSE function
    end
    cone.barfun = barfun
    cone.diffres = DiffResults.HessianResult(cone.grad)
    return
end

get_nu(cone::EpiPerSumExp) = 3

function set_initial_point(arr::AbstractVector{T}, cone::EpiPerSumExp{T}) where {T <: HypReal}
    @. arr = -log(T(cone.dim - 2))
    arr[1] = 2
    arr[2] = 1
    return arr
end

function update_feas(cone::EpiPerSumExp)
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    cone.is_feas = u > 0 && v > 0 && u > v * sum(wi -> exp(wi / v), w) # TODO use the numerically safer way to evaluate LSE function
    cone.feas_updated = true
    return cone.is_feas
end

# TODO check if this is most efficient way to use DiffResults
function update_grad(cone::EpiPerSumExp)
    @assert cone.is_feas
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.grad .= DiffResults.gradient(cone.diffres)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiPerSumExp)
    @assert cone.grad_updated
    cone.hess.data .= DiffResults.hessian(cone.diffres)
    cone.hess_updated = true
    return cone.hess
end

function update_inv_hess_prod(cone::EpiPerSumExp)
    @assert cone.hess_updated
    copyto!(cone.tmp_hess, cone.hess)
    cone.hess_fact = hyp_chol!(cone.tmp_hess)
    cone.inv_hess_prod_updated = true
    return
end

function update_inv_hess(cone::EpiPerSumExp)
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    cone.inv_hess = Symmetric(inv(cone.hess_fact), :U)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

# TODO maybe write using linear operator form rather than needing explicit hess
function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSumExp)
    @assert cone.hess_updated
    return mul!(prod, cone.hess, arr)
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiPerSumExp)
    @assert cone.inv_hess_prod_updated
    return ldiv!(prod, cone.hess_fact, arr)
end
