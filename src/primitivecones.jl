
#=
predefined standard primitive cone types
=#

abstract type PrimitiveCone end

# nonnegative orthant cone
mutable struct NonnegCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}

    function NonnegCone(dim::Int)
        prm = new()
        prm.dim = dim
        return prm
    end
end

dimension(prm::NonnegCone) = prm.dim
barrierpar_prm(prm::NonnegCone) = prm.dim
loadpnt_prm!(prm::NonnegCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)
incone_prm(prm::NonnegCone) = all(x -> (x > 0.0), prm.pnt)
calcg_prm!(g::AbstractVector{Float64}, prm::NonnegCone) = (g .= inv.(prm.pnt) .* -1.0)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::NonnegCone) = (prod .= abs2.(prm.pnt) .* arr)

# polynomial (weighted) sum of squares cone (parametrized by ip and ipwt)
mutable struct SumOfSquaresCone <: PrimitiveCone
    dim::Int
    ipwt::Vector{Matrix{Float64}}
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    F::Cholesky{Float64,Array{Float64,2}}
    ipwtpnt::Vector{Matrix{Float64}}
    Vp::Vector{Matrix{Float64}}
    Vp2::Matrix{Float64}

    function SumOfSquaresCone(dim::Int, ipwt::Vector{Matrix{Float64}})
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == dim
        end
        prm = new()
        prm.dim = dim
        prm.ipwt = ipwt
        prm.g = similar(ipwt[1], dim)
        prm.H = similar(ipwt[1], dim, dim)
        prm.ipwtpnt = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        prm.Vp = [similar(ipwt[1], dim, size(ipwtj, 2)) for ipwtj in ipwt]
        prm.Vp2 = similar(ipwt[1], dim, dim)
        return prm
    end
end

dimension(prm::SumOfSquaresCone) = prm.dim
barrierpar_prm(prm::SumOfSquaresCone) = sum(size(ipwtj, 2) for ipwtj in prm.ipwt)
loadpnt_prm!(prm::SumOfSquaresCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::SumOfSquaresCone)
    prm.g .= 0.0
    prm.H .= 0.0

    for j in eachindex(prm.ipwt) # TODO can do this loop in parallel (use separate Vp2[j])
        # prm.ipwtpnt[j] = prm.ipwt[j]'*Diagonal(prm.pnt)*prm.ipwt[j]
        mul!(prm.Vp[j], Diagonal(prm.pnt), prm.ipwt[j])
        mul!(prm.ipwtpnt[j], prm.ipwt[j]', prm.Vp[j])

        F = cholesky!(Symmetric(prm.ipwtpnt[j]), check=false)
        if !issuccess(F)
            return false
        end

        rdiv!(prm.ipwt[j], F.U)
        mul!(prm.Vp2, prm.ipwt[j], prm.ipwt[j]') # TODO if parallel, need to use separate Vp2[j]

        for i in eachindex(prm.g)
            @inbounds prm.g[i] -= prm.Vp2[i,i]
        end
        prm.H .+= abs2.(prm.Vp2)
    end

    prm.F = cholesky!(prm.H, check=false)
    if !issuccess(prm.F)
        return false
    end
    return true
end

calcg_prm!(g::AbstractVector{Float64}, prm::SumOfSquaresCone) = (g .= prm.g)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::SumOfSquaresCone) = ldiv!(prod, prm.F, arr)
