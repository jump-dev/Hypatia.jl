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

mutable struct EpiPerSumExp <: PrimitiveCone
    usedual::Bool
    dim::Int
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function EpiPerSumExp(dim::Int, isdual::Bool)
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.dim = dim
        prmtv.g = Vector{Float64}(undef, dim)
        prmtv.H = similar(prmtv.g, dim, dim)
        prmtv.H2 = similar(prmtv.H)
        function barfun(pnt)
            u = pnt[1]
            v = pnt[2]
            w = view(pnt, 3:dim)
            # return -log(u - v*sum(wi -> exp(wi/v), w)) - log(u) - log(v)
            return -log(log(u) - log(v) - log(sum(wi -> exp(wi/v), w))) - log(u) - 2.0*log(v) # TODO use the numerically safer way to evaluate LSE function
        end
        prmtv.barfun = barfun
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

EpiPerSumExp(dim::Int) = EpiPerSumExp(dim, false)

dimension(prmtv::EpiPerSumExp) = prmtv.dim
barrierpar_prmtv(prmtv::EpiPerSumExp) = 3
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::EpiPerSumExp) = (@. arr = -log(prmtv.dim - 2); arr[1] = 2.0; arr[2] = 1.0; arr)
loadpnt_prmtv!(prmtv::EpiPerSumExp, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::EpiPerSumExp)
    pnt = prmtv.pnt
    u = pnt[1]
    v = pnt[2]
    w = view(prmtv.pnt, 3:prmtv.dim)
    if u <= 0.0 || v <= 0.0 || u <= v*sum(wi -> exp(wi/v), w) # TODO use the numerically safer way to evaluate LSE function
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

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::EpiPerSumExp) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EpiPerSumExp) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::EpiPerSumExp) = mul!(prod, prmtv.H, arr)
