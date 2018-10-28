#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points ipwt

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO can perform loop for calculating g and H in parallel
TODO maybe can avoid final factorization?
TODO scale the interior direction
=#

mutable struct WSOSPolyInterp <: PrimitiveCone
    usedual::Bool
    dim::Int
    ipwt::Vector{Matrix{Float64}}
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    tmp1::Vector{Matrix{Float64}}
    tmp2::Vector{Matrix{Float64}}
    tmp3::Matrix{Float64}

    function WSOSPolyInterp(dim::Int, ipwt::Vector{Matrix{Float64}}, isdual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == dim
        end
        prmtv = new()
        prmtv.usedual = !isdual # using dual barrier
        prmtv.dim = dim
        prmtv.ipwt = ipwt
        prmtv.g = similar(ipwt[1], dim)
        prmtv.H = similar(ipwt[1], dim, dim)
        prmtv.H2 = similar(prmtv.H)
        prmtv.tmp1 = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        prmtv.tmp2 = [similar(ipwt[1], size(ipwtj, 2), dim) for ipwtj in ipwt]
        prmtv.tmp3 = similar(ipwt[1], dim, dim)
        return prmtv
    end
end

WSOSPolyInterp(dim::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterp(dim, ipwt, false)

dimension(prmtv::WSOSPolyInterp) = prmtv.dim
barrierpar_prmtv(prmtv::WSOSPolyInterp) = sum(size(ipwtj, 2) for ipwtj in prmtv.ipwt)
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::WSOSPolyInterp) = (@. arr = 1.0; arr)
loadpnt_prmtv!(prmtv::WSOSPolyInterp, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::WSOSPolyInterp)
    @. prmtv.g = 0.0
    @. prmtv.H = 0.0
    tmp3 = prmtv.tmp3

    for j in eachindex(prmtv.ipwt) # TODO can be done in parallel, but need multiple tmp3s
        ipwtj = prmtv.ipwt[j]
        tmp1j = prmtv.tmp1[j]
        tmp2j = prmtv.tmp2[j]

        # tmp1j = ipwtj'*Diagonal(pnt)*ipwtj
        mul!(tmp2j, ipwtj', Diagonal(prmtv.pnt))
        mul!(tmp1j, tmp2j, ipwtj)

        # # cholesky-based
        # F = cholesky!(Symmetric(tmp1j), Val(false), check=false) # TODO try pivoted cholesky here, but need to reorder F.U
        # if !isposdef(F)
        #     return false
        # end
        # @. tmp2j = ipwtj'
        # ldiv!(F.L, tmp2j)
        # mul!(tmp3, tmp2j', tmp2j)

        # bunch-kaufman-based
        F = bunchkaufman!(Symmetric(tmp1j), true, check=false) # TODO remove allocations (use lower-level functions here)
        if !issuccess(F)
            return false
        end
        # ipwtj * (tmp1j^-1 * ipwtj')
        @. tmp2j = ipwtj'
        ldiv!(F, tmp2j)
        mul!(tmp3, ipwtj, tmp2j)

        for i in eachindex(prmtv.g)
            prmtv.g[i] -= tmp3[i,i]
        end
        @. prmtv.H += abs2(tmp3)
    end

    @. prmtv.H2 = prmtv.H
    prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
    return issuccess(prmtv.F)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::WSOSPolyInterp) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterp) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::WSOSPolyInterp) = mul!(prod, prmtv.H, arr)
