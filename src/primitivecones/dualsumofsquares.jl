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
    Vp::Vector{Matrix{Float64}}
    Vp2::Matrix{Float64}

    function DualSumOfSquaresCone(dim::Int, ipwt::Vector{Matrix{Float64}})
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == dim
        end
        prm = new()
        prm.dim = dim
        prm.ipwt = ipwt
        prm.g = similar(ipwt[1], dim)
        prm.H = similar(ipwt[1], dim, dim)
        prm.H2 = copy(prm.H)
        prm.ipwtpnt = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        prm.Vp = [similar(ipwt[1], dim, size(ipwtj, 2)) for ipwtj in ipwt]
        prm.Vp2 = similar(ipwt[1], dim, dim)
        return prm
    end
end

dimension(prm::DualSumOfSquaresCone) = prm.dim
barrierpar_prm(prm::DualSumOfSquaresCone) = sum(size(ipwtj, 2) for ipwtj in prm.ipwt)
getintdir_prm!(arr::AbstractVector{Float64}, prm::DualSumOfSquaresCone) = (@. arr = 1.0; arr)
loadpnt_prm!(prm::DualSumOfSquaresCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::DualSumOfSquaresCone)
    @. prm.g = 0.0
    @. prm.H = 0.0

    for j in eachindex(prm.ipwt) # TODO can do this loop in parallel (use separate Vp2[j])
        # prm.ipwtpnt[j] = prm.ipwt[j]'*Diagonal(prm.pnt)*prm.ipwt[j]
        mul!(prm.Vp[j], Diagonal(prm.pnt), prm.ipwt[j])
        mul!(prm.ipwtpnt[j], prm.ipwt[j]', prm.Vp[j])

        F = cholesky!(Symmetric(prm.ipwtpnt[j]), check=false)
        if !issuccess(F)
            return false
        end

        @. prm.Vp[j] = prm.ipwt[j] # TODO this shouldn't be necessary if don't have to use rdiv
        rdiv!(prm.Vp[j], F.U) # TODO make sure this dispatches to a fast method
        mul!(prm.Vp2, prm.Vp[j], prm.Vp[j]') # TODO if parallel, need to use separate Vp2[j]

        for i in eachindex(prm.g)
            prm.g[i] -= prm.Vp2[i,i]
        end
        @. prm.H += abs2(prm.Vp2)
    end

    @. prm.H2 = prm.H
    prm.F = cholesky!(Symmetric(prm.H2), check=false) # bunchkaufman if it fails
    if !issuccess(prm.F)
        @. prm.H2 = prm.H
        prm.F = bunchkaufman!(Symmetric(prm.H2), check=false)
    end
    return issuccess(prm.F)
end

calcg_prm!(g::AbstractVector{Float64}, prm::DualSumOfSquaresCone) = (@. g = prm.g; g)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::DualSumOfSquaresCone) = ldiv!(prod, prm.F, arr)
calcHarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::DualSumOfSquaresCone) = mul!(prod, prm.H, arr)
