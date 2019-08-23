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

    lse::T
    uvlse::T
    sumwexp::T
    sumexp::T
    dzdv::T
    dzdw::Vector{T}
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact
    hess_fact_cache

    function EpiPerExp{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
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

    cone.dzdw = zeros(T, dim - 2)
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
    if u > 0 && v > 0
        cone.lse = log(sum(wi -> exp(wi / v), w))
        cone.uvlse = log(u) - log(v) - cone.lse
        cone.is_feas = cone.uvlse > 0
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiPerExp)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    lse = cone.lse
    sumexp = sum(wi -> exp(wi / v), w)
    cone.sumexp = sumexp
    sumwexp = sum(wi -> wi * exp(wi / v), w)
    cone.sumwexp = sumwexp
    uvlse = cone.uvlse

    # derivative of uvlse wrt v
    dzdv = (sumwexp / v / sumexp - 1) / v
    cone.dzdv = dzdv

    cone.grad[1] = -(inv(uvlse) + 1) / u
    cone.grad[2] = -dzdv / uvlse - 2 / v
    @. cone.dzdw = exp(w / v) / v / sumexp
    @. cone.grad[3:end] = cone.dzdw / uvlse

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiPerExp)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    lse = cone.lse
    uvlse = cone.uvlse
    sumwexp = cone.sumwexp
    dzdv = cone.dzdv
    dzdw = cone.dzdw
    sumexp = cone.sumexp
    sumwexpvv = sumwexp / v / v
    sumwsqrexpvv = sum(wi -> abs2(wi) * exp(wi / v), w) / v / v

    sumwexp = sum(wi -> wi * exp(wi / v), w) # TODO cache

    H = cone.hess.data
    H[1, 1] = (inv(abs2(uvlse)) + inv(uvlse) + 1) / u / u
    H[1, 2] = dzdv / uvlse / uvlse / u
    @views @. H[1, 3:cone.dim] = -dzdw / u / uvlse / uvlse

    # the derivative of sum(w / v ^ 2 .* exp.(w / v)) with respect to v, unscaled by inv(v ^ 2)
    dsumwexpvvdv = -(sumwsqrexpvv + 2 * sumwexp / v)
    H[2, 2] = -(dzdv / uvlse + 1 / v) / v
    H[2, 2] += (sumwexp * (dzdv / uvlse - sumwexpvv / sumexp) - dsumwexpvvdv) / v / v / sumexp
    H[2, 2] /= uvlse
    H[2, 2] += 2 / v / v

    for i in eachindex(w)
        # derivative of inv(z) wrt w
        dzidw = -dzdw[i] / uvlse / uvlse
        dsumwexpvvdw = exp(w[i] / v) * (1 + w[i] / v) / v / v
        # derivative of inv(z) * inv(v) wrt w
        H[2, 2 + i] = -dzidw / v
        # product rule for inv(z) * inv(sumexp) * sumwexpvv
        H[2, 2 + i] += dzidw * sumwexpvv / sumexp
        H[2, 2 + i] += (dzdw[i] * sumwexpvv - dsumwexpvvdw) / uvlse / sumexp
    end

    for j in eachindex(w)
        j2 = j + 2
        for i in 1:j
            H[2 + i, j2] = dzdw[i] * dzdw[j] * (inv(abs2(uvlse)) - inv(uvlse))
        end
        H[j2, j2] += dzdw[j] / v / uvlse
    end

    cone.hess_updated = true
    return cone.hess
end
