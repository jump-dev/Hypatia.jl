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
        prm = new()
        prm.dim = dim
        prm.ipwt = ipwt
        prm.g = similar(ipwt[1], dim)
        prm.H = similar(ipwt[1], dim, dim)
        prm.H2 = copy(prm.H)
        prm.ipwtpnt = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        prm.Vpt = [similar(ipwt[1], size(ipwtj, 2), dim) for ipwtj in ipwt]
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
        # ipwtpnt[j] = ipwt[j]'*Diagonal(pnt)*ipwt[j]
        mul!(prm.Vpt[j], prm.ipwt[j]', Diagonal(prm.pnt))
        mul!(prm.ipwtpnt[j], prm.Vpt[j], prm.ipwt[j])

        # # cholesky-based
        # F = cholesky!(Symmetric(prm.ipwtpnt[j]), Val(false), check=false) # TODO try pivoted cholesky here, but need to reorder F.U
        # if !isposdef(F)
        #     return false
        # end
        # @. prm.Vpt[j] = prm.ipwt[j]'
        # ldiv!(F.L, prm.Vpt[j])
        # mul!(prm.Vp2, prm.Vpt[j]', prm.Vpt[j])

        # bunch-kaufman-based
        F = bunchkaufman!(Symmetric(prm.ipwtpnt[j]), true, check=false) # TODO remove allocations (use lower-level functions here)
        if !issuccess(F)
            return false
        end
        # P * ((P'*x*P)^-1 * P')
        @. prm.Vpt[j] = prm.ipwt[j]'
        ldiv!(F, prm.Vpt[j])
        mul!(prm.Vp2, prm.ipwt[j], prm.Vpt[j])

        for i in eachindex(prm.g)
            prm.g[i] -= prm.Vp2[i,i]
        end
        @. prm.H += abs2(prm.Vp2)
    end

    @. prm.H2 = prm.H
    prm.F = cholesky!(Symmetric(prm.H2), Val(true), check=false) # bunchkaufman if it fails
    if !isposdef(prm.F)
        @. prm.H2 = prm.H
        prm.F = bunchkaufman!(Symmetric(prm.H2), true, check=false)
        return issuccess(prm.F)
    end
    return true
end

calcg_prm!(g::AbstractVector{Float64}, prm::DualSumOfSquaresCone) = (@. g = prm.g; g)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::DualSumOfSquaresCone) = ldiv!(prod, prm.F, arr)
calcHarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::DualSumOfSquaresCone) = mul!(prod, prm.H, arr)
