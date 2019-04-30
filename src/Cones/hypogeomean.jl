#=
Copyright 2018, Chris Coey and contributors

hypograph of generalized geomean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

dual barrier (modified by reflecting around u = 0 and using dual cone definition) from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i((w_i/alpha_i)^alpha_i) + u) - sum_i((1 - alpha_i)*log(w_i/alpha_i)) - log(-u)

TODO try to make barrier evaluation more efficient
=#

mutable struct HypoGeomean <: Cone
    use_dual::Bool
    dim::Int
    alpha::Vector{Float64}

    ialpha::Vector{Float64}
    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function HypoGeomean(alpha::Vector{Float64}, is_dual::Bool)
        dim = length(alpha) + 1
        @assert dim >= 3
        @assert all(ai >= 0.0 for ai in alpha)
        @assert sum(alpha) ≈ 1.0 atol = 1e-9
        cone = new()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.alpha = alpha
        return cone
    end
end

HypoGeomean(alpha::Vector{Float64}) = HypoGeomean(alpha, false)

function setup_data(cone::HypoGeomean)
    dim = cone.dim
    alpha = cone.alpha
    ialpha = inv.(alpha)
    cone.ialpha = ialpha
    cone.g = Vector{Float64}(undef, dim)
    cone.H = Matrix{Float64}(undef, dim, dim)
    cone.H2 = similar(cone.H)
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

set_initial_point(arr::AbstractVector{Float64}, cone::HypoGeomean) = (@. arr = 1.0; arr[1] = -prod(cone.ialpha[i]^cone.alpha[i] for i in eachindex(cone.alpha)) / cone.dim; arr)

function check_in_cone(cone::HypoGeomean)
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    if u >= 0.0 || any(wi <= 0.0 for wi in w)
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
