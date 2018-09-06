
# second order cone
mutable struct SecondOrderCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    dist::Float64
    Hi::Matrix{Float64}

    function SecondOrderCone(dim::Int)
        prm = new()
        prm.dim = dim
        prm.Hi = Matrix{Float64}(undef, dim, dim)
        return prm
    end
end

dimension(prm::SecondOrderCone) = prm.dim
barrierpar_prm(prm::SecondOrderCone) = 1.0
getintdir_prm!(arr::AbstractVector{Float64}, prm::SecondOrderCone) = (arr[1] = 1.0; arr[2:end] .= 0.0; arr)
loadpnt_prm!(prm::SecondOrderCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::SecondOrderCone)
    if prm.pnt[1] <= 0
        return false
    end

    prm.dist = abs2(prm.pnt[1]) - sum(abs2, prm.pnt[j] for j in 2:prm.dim)
    if prm.dist > 0.0
        mul!(prm.Hi, prm.pnt, prm.pnt')
        prm.Hi .+= prm.Hi
        prm.Hi[1,1] -= prm.dist
        for j in 2:prm.dim
            prm.Hi[j,j] += prm.dist
        end
        return true
    end
    return false
end

calcg_prm!(g::AbstractVector{Float64}, prm::SecondOrderCone) = (g .= inv(prm.dist) .* prm.pnt; g[1] *= -1.0; g)
# TODO if later use Linv instead of Hinv, see Vandenberghe coneprog.dvi for analytical Linv
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::SecondOrderCone) = mul!(prod, prm.Hi, arr)
