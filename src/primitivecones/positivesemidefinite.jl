#=
Copyright 2018, Chris Coey and contributors

positive semidefinite cone lower triangle, svec (scaled) definition
barrier for matrix cone is -ln det(X)
from Nesterov & Todd "Self-Scaled Barriers and Interior-Point Methods for Convex Programming"
=#

mutable struct PositiveSemidefiniteCone <: PrimitiveCone
    dim::Int
    side::Int
    pnt::AbstractVector{Float64}
    mat::Matrix{Float64}
    mat2::Matrix{Float64}
    matpnt::Matrix{Float64}
    matinv::Matrix{Float64}

    function PositiveSemidefiniteCone(dim::Int)
        prmtv = new()
        prmtv.dim = dim
        prmtv.side = round(Int, sqrt(0.25 + dim + dim) - 0.5)
        prmtv.mat = Matrix{Float64}(undef, prmtv.side, prmtv.side)
        prmtv.mat2 = copy(prmtv.mat)
        prmtv.matpnt = copy(prmtv.mat)
        return prmtv
    end
end

dimension(prmtv::PositiveSemidefiniteCone) = prmtv.dim
barrierpar_prmtv(prmtv::PositiveSemidefiniteCone) = prmtv.side
function getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::PositiveSemidefiniteCone)
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
loadpnt_prmtv!(prmtv::PositiveSemidefiniteCone, pnt::AbstractVector{Float64}) = (prmtv.pnt = pnt)

function incone_prmtv(prmtv::PositiveSemidefiniteCone)
    vectomat!(prmtv.mat, prmtv.pnt)
    F = cholesky!(Symmetric(prmtv.mat), check=false)
    if !issuccess(F)
        return false
    end

    prmtv.matinv = -inv(F) # TODO eliminate allocs
    vectomat!(prmtv.matpnt, prmtv.pnt)
    return true
end

calcg_prmtv!(g::AbstractVector{Float64}, prmtv::PositiveSemidefiniteCone) = (mattovec!(g, prmtv.matinv); g)

function calcHiarr_prmtv!(prod::AbstractVector{Float64}, arr::AbstractVector{Float64}, prmtv::PositiveSemidefiniteCone)
    vectomat!(prmtv.mat, arr)
    mul!(prmtv.mat2, prmtv.mat, prmtv.matpnt)
    mul!(prmtv.mat, prmtv.matpnt, prmtv.mat2)
    mattovec!(prod, prmtv.mat)
    return prod
end

function calcHiarr_prmtv!(prod::AbstractMatrix{Float64}, arr::AbstractMatrix{Float64}, prmtv::PositiveSemidefiniteCone)
    for j in 1:size(arr, 2)
        vectomat!(prmtv.mat, view(arr, :, j))
        mul!(prmtv.mat2, prmtv.mat, prmtv.matpnt)
        mul!(prmtv.mat, prmtv.matpnt, prmtv.mat2)
        mattovec!(view(prod, :, j), prmtv.mat)
    end
    return prod
end

function calcHarr_prmtv!(prod::AbstractVector{Float64}, arr::AbstractVector{Float64}, prmtv::PositiveSemidefiniteCone)
    vectomat!(prmtv.mat, arr)
    mul!(prmtv.mat2, prmtv.mat, prmtv.matinv)
    mul!(prmtv.mat, prmtv.matinv, prmtv.mat2)
    mattovec!(prod, prmtv.mat)
    return prod
end

function calcHarr_prmtv!(prod::AbstractMatrix{Float64}, arr::AbstractMatrix{Float64}, prmtv::PositiveSemidefiniteCone)
    for j in 1:size(arr, 2)
        vectomat!(prmtv.mat, view(arr, :, j))
        mul!(prmtv.mat2, prmtv.mat, prmtv.matinv)
        mul!(prmtv.mat, prmtv.matinv, prmtv.mat2)
        mattovec!(view(prod, :, j), prmtv.mat)
    end
    return prod
end
