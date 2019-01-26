#=
Copyright 2018, Chris Coey and contributors

row-wise lower triangle (svec space) of positive semidefinite matrix cone
(smat space) W \in S^n : 0 >= eigmin(W)
(see equivalent MathOptInterface PositiveSemidefiniteConeTriangle definition)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-logdet(W)

TODO eliminate allocations for inverse-finding
=#

mutable struct PosSemidef <: Cone
    usedual::Bool
    dim::Int
    side::Int
    point::AbstractVector{Float64}
    mat::Matrix{Float64}
    mat2::Matrix{Float64}
    matpnt::Matrix{Float64}
    matinv::Matrix{Float64}

    function PosSemidef(dim::Int, isdual::Bool)
        cone = new()
        cone.usedual = isdual
        cone.dim = dim
        cone.side = round(Int, sqrt(0.25 + 2 * dim) - 0.5)
        cone.mat = Matrix{Float64}(undef, cone.side, cone.side)
        cone.mat2 = similar(cone.mat)
        cone.matpnt = similar(cone.mat)
        return cone
    end
end

PosSemidef(dim::Int) = PosSemidef(dim, false)

get_nu(cone::PosSemidef) = cone.side

function set_initial_point(arr::AbstractVector{Float64}, cone::PosSemidef)
    for i in 1:cone.side, j in i:cone.side
        if i == j
            cone.mat[i, j] = 1.0
        else
            cone.mat[i, j] = cone.mat[j, i] = 0.0
        end
    end
    smat_to_svec!(arr, cone.mat)
    return arr
end

function check_in_cone(cone::PosSemidef)
    svec_to_smat!(cone.mat, cone.point)
    @. cone.matpnt = cone.mat

    F = cholesky!(Symmetric(cone.mat), Val(true), check = false)
    if !isposdef(F)
        return false
    end
    cone.matinv = -inv(F) # TODO eliminate allocs
    return true
end



calcg!(g::AbstractVector{Float64}, cone::PosSemidef) = (smat_to_svec!(g, cone.matinv); g)

function calcHiarr!(prod::AbstractVector{Float64}, arr::AbstractVector{Float64}, cone::PosSemidef)
    svec_to_smat!(cone.mat, arr)
    mul!(cone.mat2, cone.mat, cone.matpnt)
    mul!(cone.mat, cone.matpnt, cone.mat2)
    smat_to_svec!(prod, cone.mat)
    return prod
end

function calcHiarr!(prod::AbstractMatrix{Float64}, arr::AbstractMatrix{Float64}, cone::PosSemidef)
    for j in 1:size(arr, 2)
        svec_to_smat!(cone.mat, view(arr, :, j))
        mul!(cone.mat2, cone.mat, cone.matpnt)
        mul!(cone.mat, cone.matpnt, cone.mat2)
        smat_to_svec!(view(prod, :, j), cone.mat)
    end
    return prod
end

function calcHarr!(prod::AbstractVector{Float64}, arr::AbstractVector{Float64}, cone::PosSemidef)
    svec_to_smat!(cone.mat, arr)
    mul!(cone.mat2, cone.mat, cone.matinv)
    mul!(cone.mat, cone.matinv, cone.mat2)
    smat_to_svec!(prod, cone.mat)
    return prod
end

function calcHarr!(prod::AbstractMatrix{Float64}, arr::AbstractMatrix{Float64}, cone::PosSemidef)
    for j in 1:size(arr, 2)
        svec_to_smat!(cone.mat, view(arr, :, j))
        mul!(cone.mat2, cone.mat, cone.matinv)
        mul!(cone.mat, cone.matinv, cone.mat2)
        smat_to_svec!(view(prod, :, j), cone.mat)
    end
    return prod
end
