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

mutable struct EpiPerPower <: PrimitiveCone
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
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.alpha = alpha
        prmtv.g = Vector{Float64}(undef, 3)
        prmtv.H = similar(prmtv.g, 3, 3)
        prmtv.H2 = similar(prmtv.H)
        ialpha2 = 2.0*inv(alpha)
        function barfun(pnt)
            (u, v, w) = (pnt[1], pnt[2], pnt[3])
            if alpha >= 2.0
                return -log(u*v^(2.0 - ialpha2) - abs2(w)*u^(1.0 - ialpha2))
            else
                return -log(u^ialpha2*v - abs2(w)*v^(ialpha2 - 1.0))
            end
        end
        prmtv.barfun = barfun
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

EpiPerPower(alpha::Vector{Float64}) = EpiPerPower(alpha, false)

dimension(prmtv::EpiPerPower) = 3
barrierpar_prmtv(prmtv::EpiPerPower) = 3 - 2*min(inv(prmtv.alpha), 1 - inv(prmtv.alpha))
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::EpiPerPower) = (arr[1] = 1.0; arr[2] = 1.0; arr[3] = 0.0; arr)
loadpnt_prmtv!(prmtv::EpiPerPower, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::EpiPerPower)
    u = prmtv.pnt[1]
    v = prmtv.pnt[2]
    w = prmtv.pnt[3]
    alpha = prmtv.alpha
    if u <= 0.0 || v <= 0.0 || u < v*(abs(w/v))^alpha
        return false
    end

    # TODO check allocations, check with Jarrett if this is most efficient way to use DiffResults
    prmtv.diffres = ForwardDiff.hessian!(prmtv.diffres, prmtv.barfun, prmtv.pnt)
    prmtv.g .= DiffResults.gradient(prmtv.diffres)
    prmtv.H .= DiffResults.hessian(prmtv.diffres)

    @. prmtv.H2 = prmtv.H
    prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
    return issuccess(prmtv.F)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::EpiPerPower) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EpiPerPower) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EpiPerPower) = mul!(prod, prmtv.H, arr)
