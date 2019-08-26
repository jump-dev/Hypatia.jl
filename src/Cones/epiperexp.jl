#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

(closure of) epigraph of perspective of sum of exponentials (n-dimensional exponential cone)
(u in R, v in R_+, w in R^n) : u >= v*sum(exp.(w/v))

barrier (guessed)
-log(v*log(u/v) - v*logsumexp(w/v)) - log(u) - log(v)
in the three dimensional case this matches the barrier for hypoperlog (self-concordant)
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
    expwv::Vector{T}
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
    cone.expwv = zeros(T, dim - 2)
    cone.dzdw = zeros(T, dim - 2)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.hess_fact_cache = nothing
    return
end

get_nu(cone::EpiPerExp) = 3

function set_initial_point(arr::AbstractVector{T}, cone::EpiPerExp{T}) where {T <: Real}
    @. arr = -log(T(cone.dim - 2))
    arr[1] = exp(1)
    arr[2] = 1
    return arr
end

function update_feas(cone::EpiPerExp)
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    if u > 0 && v > 0
        @. cone.expwv = exp(w / v)
        cone.sumexp = sum(cone.expwv)
        cone.lse = log(cone.sumexp)
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
    sumexp = cone.sumexp
    sumwexp = dot(w, cone.expwv)
    cone.sumwexp = sumwexp
    uvlse = cone.uvlse
    cone.dzdv = (sumwexp / v / sumexp - 1) / v # derivative of uvlse wrt v
    cone.grad[1] = -(inv(uvlse) + 1) / u
    cone.grad[2] = -cone.dzdv / uvlse - 2 / v
    @. cone.dzdw = cone.expwv / v / sumexp
    @. cone.grad[3:end] = cone.dzdw / uvlse
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::EpiPerExp)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    expwv = cone.expwv
    lse = cone.lse
    uvlse = cone.uvlse
    sumwexp = cone.sumwexp
    dzdv = cone.dzdv
    dzdw = cone.dzdw
    sumexp = cone.sumexp
    H = cone.hess.data

    H[1, 1] = ((inv(uvlse) + 1) / uvlse + 1) / u / u
    H[1, 2] = dzdv / uvlse / uvlse / u
    @. @views H[1, 3:cone.dim] = -dzdw / u / uvlse / uvlse

    # the derivative of sum(w / v ^ 2 .* exp.(w / v)) with respect to v, unscaled by inv(v ^ 2)
    sumwexpvv = sumwexp / v / v
    sumwsqrexpvv = sum(abs2(w[j]) * expwv[j] for j in eachindex(w)) / v / v
    dsumwexpvvdv = -(sumwsqrexpvv + 2 * sumwexp / v)
    H[2, 2] = -(dzdv / uvlse + 1 / v) / v
    H[2, 2] += (sumwexp * (dzdv / uvlse - sumwexpvv / sumexp) - dsumwexpvvdv) / v / v / sumexp
    H[2, 2] /= uvlse
    H[2, 2] += 2 / v / v

    for i in eachindex(w)
        # derivative of inv(z) wrt w
        dzidw = -dzdw[i] / uvlse / uvlse
        dsumwexpvvdw = expwv[i] * (1 + w[i] / v) / v / v
        # derivative of inv(z) * inv(v) wrt w
        H[2, 2 + i] = -dzidw / v
        # product rule for inv(z) * inv(sumexp) * sumwexpvv
        H[2, 2 + i] += dzidw * sumwexpvv / sumexp
        H[2, 2 + i] += (dzdw[i] * sumwexpvv - dsumwexpvvdw) / uvlse / sumexp
        # equivalently H[2, 2 + i] += ((dzdw[i] * sumwexpvv * (1 - inv(uvlse)) - dsumwexpvvdw) / uvlse) / sumexp
    end

    for j in eachindex(w)
        j2 = j + 2
        for i in 1:j
            H[2 + i, j2] = dzdw[i] * dzdw[j] * (inv(uvlse) - 1) / uvlse
        end
        H[j2, j2] += dzdw[j] / v / uvlse
    end

    cone.hess_updated = true
    return cone.hess
end
