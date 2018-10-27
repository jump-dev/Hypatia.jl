#=
Copyright 2018, Chris Coey and contributors

hypograph of generalized geomean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

barrier *modified* from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i(w_i^alpha_i) - u) - sum_i((1 - alpha_i)*log(w_i))

TODO check if this modified barrier remains self-concordant log-homogeneous or not (can it be modified if not?)
=#

mutable struct HypoGeomean <: PrimitiveCone
    dim::Int
    alpha::Vector{Float64}
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    barfun::Function
    diffres

    function HypoGeomean(alpha::Vector{Float64})
        dim = length(alpha) + 1
        @assert dim >= 3
        @assert all(ai >= 0.0 for ai in alpha)
        @assert sum(alpha) == 1.0
        prmtv = new()
        prmtv.dim = dim
        prmtv.alpha = alpha
        prmtv.g = Vector{Float64}(undef, dim)
        prmtv.H = similar(prmtv.g, dim, dim)
        prmtv.H2 = similar(prmtv.H)
        function barfun(pnt)
            u = pnt[1]
            w = view(pnt, 2:dim)
            return -log(prod(w[i]^alpha[i] for i in eachindex(alpha)) - u) - sum((1.0 - alpha[i])*log(w[i]) for i in eachindex(alpha))
        end
        prmtv.barfun = barfun
        prmtv.diffres = DiffResults.HessianResult(prmtv.g)
        return prmtv
    end
end

dimension(prmtv::HypoGeomean) = prmtv.dim
barrierpar_prmtv(prmtv::HypoGeomean) = prmtv.dim - 1
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::HypoGeomean) = (@. arr = 1.0; arr[1] = 0.0; arr)
loadpnt_prmtv!(prmtv::HypoGeomean, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::HypoGeomean)
    u = prmtv.pnt[1]
    w = view(prmtv.pnt, 2:prmtv.dim)
    alpha = prmtv.alpha
    if any(wi <= 0.0 for wi in w)
        return false
    end
    if prod(w[i]^alpha[i] for i in eachindex(alpha)) <= u
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

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::HypoGeomean) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::HypoGeomean) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::HypoGeomean) = mul!(prod, prmtv.H, arr)
