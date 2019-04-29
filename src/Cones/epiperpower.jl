#=
Copyright 2018, Chris Coey and contributors

epigraph of perspective of convex power of absolute value function (AKA 3-dim power cone) parametrized by real alpha > 1
(u in R, v in R_+, w in R) : u >= v*|w/v|^alpha
equivalent to u >= v^(1-alpha)*|w|^alpha or u^(1/alpha)*v^(1-1/alpha) >= |w|

barrier from "Cones and Interior-Point Algorithms for Structured Convex Optimization involving Powers and Exponentials" by P. Chares 2007
-log(u^(2/alpha)*v^(2-2/alpha) - w^2) - max{1-2/alpha, 0}*log(u) - max{2/alpha-1, 0}*log(v)

TODO get gradient and hessian analytically (may be nicer if redefine as u >= v/alpha*|w/v|^alpha)
TODO although this barrier has a lower parameter, maybe the more standard barrier is more numerically robust
=#

mutable struct EpiPerPower <: Cone
    use_dual::Bool
    alpha::Float64

    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function EpiPerPower(alpha::Float64, is_dual::Bool)
        @assert alpha > 1.0
        cone = new()
        cone.use_dual = is_dual
        cone.alpha = alpha
        return cone
    end
end

EpiPerPower(alpha::Float64) = EpiPerPower(alpha, false)

function setup_data(cone::EpiPerPower)
    cone.g = Vector{Float64}(undef, 3)
    cone.H = Matrix{Float64}(undef, 3, 3)
    cone.H2 = similar(cone.H)

    alpha = cone.alpha
    ialpha2 = 2.0 * inv(alpha)
    if alpha >= 2.0
        cone.barfun = point -> -log(point[1] * point[2]^(2.0 - ialpha2) - abs2(point[3]) * point[1]^(1.0 - ialpha2))
    else
        cone.barfun = point -> -log(point[1]^ialpha2 * point[2] - abs2(point[3]) * point[2]^(ialpha2 - 1.0))
    end
    cone.diffres = DiffResults.HessianResult(cone.g)
    return
end

dimension(cone::EpiPerPower) = 3

get_nu(cone::EpiPerPower) = 3 - 2 * min(inv(cone.alpha), 1.0 - inv(cone.alpha))

set_initial_point(arr::AbstractVector{Float64}, cone::EpiPerPower) = (arr[1] = 1.0; arr[2] = 1.0; arr[3] = 0.0; arr)

function check_in_cone(cone::EpiPerPower)
    (u, v, w) = cone.point
    if u <= 0.0 || v <= 0.0 || u <= v * (abs(w / v))^cone.alpha
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.g .= DiffResults.gradient(cone.diffres)
    cone.H .= DiffResults.hessian(cone.diffres)

    return factorize_hess(cone)
end
