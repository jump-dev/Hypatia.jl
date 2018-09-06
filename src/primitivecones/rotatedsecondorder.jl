
# rotated second order cone
mutable struct RotatedSecondOrderCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    Hi::Matrix{Float64}

    function RotatedSecondOrderCone(dim::Int)
        prm = new()
        prm.dim = dim
        prm.g = Vector{Float64}(undef, dim)
        prm.Hi = Matrix{Float64}(undef, dim, dim)
        return prm
    end
end

dimension(prm::RotatedSecondOrderCone) = prm.dim
barrierpar_prm(prm::RotatedSecondOrderCone) = 2.0
getintdir_prm!(arr::AbstractVector{Float64}, prm::RotatedSecondOrderCone) = (arr[1:2] .= 1.0; arr[3:end] .= 0.0; arr)
loadpnt_prm!(prm::RotatedSecondOrderCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::RotatedSecondOrderCone)
    pnt = prm.pnt
    if (pnt[1] <= 0) || (pnt[2] <= 0)
        return false
    end

    nrm2 = 0.5*sum(abs2, pnt[j] for j in 3:prm.dim)
    disth = pnt[1]*pnt[2] - nrm2
    if disth > 0.0
        g = prm.g
        g .= pnt
        g[1] = -pnt[2]
        g[2] = -pnt[1]
        g ./= disth

        Hi = prm.Hi
        mul!(Hi, pnt, pnt')
        Hi[2,1] = Hi[1,2] = nrm2
        for j in 3:prm.dim
            Hi[j,j] += disth
        end
        return true
    end
    return false
end

calcg_prm!(g::AbstractVector{Float64}, prm::RotatedSecondOrderCone) = (g .= prm.g; g)
# TODO if later use Linv instead of Hinv, see Vandenberghe coneprog.dvi for analytical Linv
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::RotatedSecondOrderCone) = mul!(prod, prm.Hi, arr)
