#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

dual cone of the polynomial (weighted) sum of squares cone (parametrized by ipwt)
=#

mutable struct DualSumOfSquaresCone <: PrimitiveCone
    dim::Int
    ipwt::Vector{Matrix{Float64}}
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    ipwtpnt::Vector{Matrix{Float64}}
    Vpt::Vector{Matrix{Float64}}
    Vp2::Matrix{Float64}

    function DualSumOfSquaresCone(dim::Int, ipwt::Vector{Matrix{Float64}})
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == dim
        end
        prmtv = new()
        prmtv.dim = dim
        prmtv.ipwt = ipwt
        prmtv.g = similar(ipwt[1], dim)
        prmtv.H = similar(ipwt[1], dim, dim)
        prmtv.H2 = similar(prmtv.H)
        prmtv.ipwtpnt = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        prmtv.Vpt = [similar(ipwt[1], size(ipwtj, 2), dim) for ipwtj in ipwt]
        prmtv.Vp2 = similar(ipwt[1], dim, dim)
        return prmtv
    end
end

dimension(prmtv::DualSumOfSquaresCone) = prmtv.dim
barrierpar_prmtv(prmtv::DualSumOfSquaresCone) = sum(size(ipwtj, 2) for ipwtj in prmtv.ipwt)
getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::DualSumOfSquaresCone) = (@. arr = 1.0; arr)
loadpnt_prmtv!(prmtv::DualSumOfSquaresCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::DualSumOfSquaresCone)
    @. prmtv.g = 0.0
    @. prmtv.H = 0.0

    for j in eachindex(prmtv.ipwt) # TODO can do this loop in parallel (use separate Vp2[j])
        # ipwtpnt[j] = ipwt[j]'*Diagonal(pnt)*ipwt[j]
        mul!(prmtv.Vpt[j], prmtv.ipwt[j]', Diagonal(prmtv.pnt))
        mul!(prmtv.ipwtpnt[j], prmtv.Vpt[j], prmtv.ipwt[j])

        # # cholesky-based
        # F = cholesky!(Symmetric(prmtv.ipwtpnt[j]), Val(false), check=false) # TODO try pivoted cholesky here, but need to reorder F.U
        # if !isposdef(F)
        #     return false
        # end
        # @. prmtv.Vpt[j] = prmtv.ipwt[j]'
        # ldiv!(F.L, prmtv.Vpt[j])
        # mul!(prmtv.Vp2, prmtv.Vpt[j]', prmtv.Vpt[j])

        # bunch-kaufman-based
        F = bunchkaufman!(Symmetric(prmtv.ipwtpnt[j]), true, check=false) # TODO remove allocations (use lower-level functions here)
        if !issuccess(F)
            return false
        end
        # P * ((P'*x*P)^-1 * P')
        @. prmtv.Vpt[j] = prmtv.ipwt[j]'
        ldiv!(F, prmtv.Vpt[j])
        mul!(prmtv.Vp2, prmtv.ipwt[j], prmtv.Vpt[j])

        for i in eachindex(prmtv.g)
            prmtv.g[i] -= prmtv.Vp2[i,i]
        end
        @. prmtv.H += abs2(prmtv.Vp2)
    end

    @. prmtv.H2 = prmtv.H
    prmtv.F = bunchkaufman!(Symmetric(prmtv.H2), true, check=false)
    return issuccess(prmtv.F)
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::DualSumOfSquaresCone) = (@. g = prmtv.g; g)
calcHiarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::DualSumOfSquaresCone) = ldiv!(prod, prmtv.F, arr)
calcHarr_prmtv!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prmtv::DualSumOfSquaresCone) = mul!(prod, prmtv.H, arr)
