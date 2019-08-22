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
    lse::T
    uvlse::T
    sumwexpv::T
    dzdv::T
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
    cone.lse = log(sum(wi -> exp(wi / v), w))
    cone.is_feas = u > 0 && v > 0 && log(u) > cone.lse
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
    sumwexpv = sum(wi -> wi * exp(wi / v), w) / v
    cone.sumwexpv = sumwexpv
    uvlse = log(u) - log(v) - lse
    cone.uvlse = uvlse
    # derivative of uvlse wrt v
    dzdv = -inv(v) + sumwexpv / sumexp / v
    cone.dzdv = dzdv

    cone.grad[1] = -(inv(uvlse) + 1) / u
    cone.grad[2] = -dzdv / uvlse - 2 / v
    # cone.grad[2] = ((1 - sumwexpv / sumexp) / uvlse - 2) / v
    @. cone.grad[3:end] = exp(w / v) / v / sumexp / uvlse

    # cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)

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
    sumwexpv = cone.sumwexpv
    dzdv = cone.dzdv
    sumexp = sum(wi -> exp(wi / v), w) # TODO cache

    H = cone.hess.data
    H[1, 1] = (1 / uvlse / uvlse + 1 / uvlse + 1) / u / u
    H[1, 2] = dzdv / uvlse / uvlse / u

    t1 = inv(uvlse)
    t2 = inv(sumexp)
    t3(i) = w[i] / v / v
    t4(i) = exp(w[i] / v)
    t3t4 = sum(t3(i) * t4(i) for i in eachindex(w))
    dt1dv = -inv(abs2(uvlse)) * dzdv
    dt2dv = inv(abs2(sumexp)) * sumwexpv / v
    dt3dv(i) = -2 * w[i] / v / v / v
    dt4dv(i) = -w[i] / v / v * exp(w[i] / v)
    dt3t4dv = sum(t3(i) * dt4dv(i) + dt3dv(i) * t4(i) for i in eachindex(w))

    H[2, 2] = -dzdv / uvlse / uvlse / v - 1 / uvlse / v / v
    H[2, 2] += 2 / v / v
    H[2, 2] -= dt1dv * t2 * t3t4
    H[2, 2] -= t1 * (dt2dv * t3t4 + t2 * dt3t4dv)

    for i in eachindex(w)
        H[1, 2 + i] = -exp(w[i] / v) / v / sumexp / u / uvlse / uvlse
    end

    for i in eachindex(w)
        dt1dw = -inv(abs2(uvlse)) * exp(w[i] / v) / v / sumexp
        dt2dw = -inv(abs2(sumexp)) * exp(w[i] / v) / v
        dt3t4dw = exp(w[i] / v) * (1 / v / v + w[i] / v^3)
        # dt3t4dw = 1 / v^2 * exp(w[i] / v) +
        H[2, 2 + i] = inv(abs2(uvlse)) * exp(w[i] / v) / v / sumexp / v # repetition
        H[2, 2 + i] += dt1dw * t2 * t3t4
        H[2, 2 + i] -= t1 * (dt2dw * t3t4 + dt3t4dw * t2)
    end

    # dsumexpdv = -sumwexpv / v
    # for i in eachindex(w)
    #     H[2, 2 + i] += -inv(abs2(uvlse)) * cone.dzdv * inv(sumexp) * exp(w[i] * v)
    #     term2 = -inv(abs2(sumexp)) * dsumexpdv / v * exp(w[i] / v)
    #     term2 += (-sumexp / v^2 + )
    # end

    for j in eachindex(w)
        for i in 1:j
            # H[2 + i, 2 + j] = -inv(abs2(uvlse)) * inv(abs2(sumexp)) * inv(abs2(v)) * exp(w[i] / v) * exp(w[j] / v)
            H[2 + i, 2 + j] = inv(abs2(sumexp)) * inv(abs2(v)) * exp(w[i] / v) * exp(w[j] / v) * (inv(abs2(uvlse)) - inv(uvlse))
        end
        H[j + 2, j + 2] += inv(uvlse) * inv(sumexp) * inv(abs2(v)) * exp(w[j] / v)
    end

    cone.hess_updated = true
    return cone.hess
end
