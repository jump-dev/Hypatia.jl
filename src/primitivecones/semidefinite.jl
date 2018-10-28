#=
Copyright 2018, Chris Coey and contributors

row-wise lower triangle (svec space) of positive semidefinite matrix cone
(smat space) W \in S^n : 0 >= eigmin(W)
(see equivalent MathOptInterface PositiveSemidefiniteConeTriangle definition)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-logdet(W)

TODO eliminate allocations for inverse-finding
=#

mutable struct PosSemidef <: PrimitiveCone
    usedual::Bool
    dim::Int
    side::Int
    pnt::AbstractVector{Float64}
    mat::Matrix{Float64}
    mat2::Matrix{Float64}
    matpnt::Matrix{Float64}
    matinv::Matrix{Float64}

    function PosSemidef(dim::Int, isdual::Bool)
        prmtv = new()
        prmtv.usedual = isdual
        prmtv.dim = dim
        prmtv.side = round(Int, sqrt(0.25 + 2*dim) - 0.5)
        prmtv.mat = Matrix{Float64}(undef, prmtv.side, prmtv.side)
        prmtv.mat2 = similar(prmtv.mat)
        prmtv.matpnt = similar(prmtv.mat)
        return prmtv
    end
end

PosSemidef(dim::Int) = PosSemidef(dim, false)

dimension(prmtv::PosSemidef) = prmtv.dim
barrierpar_prmtv(prmtv::PosSemidef) = prmtv.side
function getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::PosSemidef)
    for i in 1:prmtv.side, j in i:prmtv.side
        if i == j
            prmtv.mat[i,j] = 1.0
        else
            prmtv.mat[i,j] = prmtv.mat[j,i] = 0.0
        end
    end
    mattovec!(arr, prmtv.mat)
    return arr
end
loadpnt_prmtv!(prmtv::PosSemidef, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::PosSemidef)
    vectomat!(prmtv.mat, prmtv.pnt)
    @. prmtv.matpnt = prmtv.mat

    F = bunchkaufman!(Symmetric(prmtv.mat), true, check=false)
    if !issuccess(F)
        return false
    end
    prmtv.matinv = -inv(F) # TODO eliminate allocs
    return true
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::PosSemidef) = (mattovec!(g, prmtv.matinv); g)

function calcHiarr_prmtv!(prod::AbstractVector{Float64}, arr::AbstractVector{Float64}, prmtv::PosSemidef)
    vectomat!(prmtv.mat, arr)
    mul!(prmtv.mat2, prmtv.mat, prmtv.matpnt)
    mul!(prmtv.mat, prmtv.matpnt, prmtv.mat2)
    mattovec!(prod, prmtv.mat)
    return prod
end

function calcHiarr_prmtv!(prod::AbstractMatrix{Float64}, arr::AbstractMatrix{Float64}, prmtv::PosSemidef)
    for j in 1:size(arr, 2)
        vectomat!(prmtv.mat, view(arr, :, j))
        mul!(prmtv.mat2, prmtv.mat, prmtv.matpnt)
        mul!(prmtv.mat, prmtv.matpnt, prmtv.mat2)
        mattovec!(view(prod, :, j), prmtv.mat)
    end
    return prod
end

function calcHarr_prmtv!(prod::AbstractVector{Float64}, arr::AbstractVector{Float64}, prmtv::PosSemidef)
    vectomat!(prmtv.mat, arr)
    mul!(prmtv.mat2, prmtv.mat, prmtv.matinv)
    mul!(prmtv.mat, prmtv.matinv, prmtv.mat2)
    mattovec!(prod, prmtv.mat)
    return prod
end

function calcHarr_prmtv!(prod::AbstractMatrix{Float64}, arr::AbstractMatrix{Float64}, prmtv::PosSemidef)
    for j in 1:size(arr, 2)
        vectomat!(prmtv.mat, view(arr, :, j))
        mul!(prmtv.mat2, prmtv.mat, prmtv.matinv)
        mul!(prmtv.mat, prmtv.matinv, prmtv.mat2)
        mattovec!(view(prod, :, j), prmtv.mat)
    end
    return prod
end
