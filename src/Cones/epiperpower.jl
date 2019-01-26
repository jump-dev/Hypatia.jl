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
    usedual::Bool
    alpha::Float64
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function EpiPerPower(alpha::Float64, isdual::Bool)
        @assert alpha > 1.0
        cone = new()
        cone.usedual = isdual
        cone.alpha = alpha
        cone.g = Vector{Float64}(undef, 3)
        cone.H = similar(cone.g, 3, 3)
        cone.H2 = similar(cone.H)
        ialpha2 = 2.0*inv(alpha)
        function barfun(pnt)
            (u, v, w) = (pnt[1], pnt[2], pnt[3])
            if alpha >= 2.0
                return -log(u*v^(2.0 - ialpha2) - abs2(w)*u^(1.0 - ialpha2))
            else
                return -log(u^ialpha2*v - abs2(w)*v^(ialpha2 - 1.0))
            end
        end
        cone.barfun = barfun
        cone.diffres = DiffResults.HessianResult(cone.g)
        return cone
    end
end

EpiPerPower(alpha::Float64) = EpiPerPower(alpha, false)

dimension(cone::EpiPerPower) = 3
get_nu(cone::EpiPerPower) = 3 - 2*min(inv(cone.alpha), 1.0 - inv(cone.alpha))
set_initial_point(arr::AbstractVector{Float64}, cone::EpiPerPower) = (arr[1] = 1.0; arr[2] = 1.0; arr[3] = 0.0; arr)
loadpnt!(cone::EpiPerPower, pnt::AbstractVector{Float64}) = (cone.pnt = pnt)

function incone(cone::EpiPerPower, scal::Float64)
    u = cone.pnt[1]
    v = cone.pnt[2]
    w = cone.pnt[3]
    alpha = cone.alpha
    if u <= 0.0 || v <= 0.0 || u <= v*(abs(w/v))^alpha
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.pnt)
    cone.g .= DiffResults.gradient(cone.diffres)
    cone.H .= DiffResults.hessian(cone.diffres)

    return factorize_hess(cone)
end
