
# positive semidefinite cone (lower triangle, off-diagonals scaled)
# barrier for matrix cone is -ln det(X)
# from Nesterov & Todd "Self-Scaled Barriers and Interior-Point Methods for Convex Programming"
mutable struct PositiveSemidefiniteCone <: PrimitiveCone
    dim::Int
    side::Int
    pnt::AbstractVector{Float64}
    mat::Matrix{Float64}
    mat2::Matrix{Float64}
    matpnt::Matrix{Float64}
    matinv::Matrix{Float64}

    function PositiveSemidefiniteCone(dim::Int)
        prm = new()
        prm.dim = dim
        prm.side = round(Int, sqrt(0.25 + dim + dim) - 0.5)
        prm.mat = Matrix{Float64}(undef, prm.side, prm.side)
        prm.mat2 = copy(prm.mat)
        prm.matpnt = copy(prm.mat)
        # prm.matinv = copy(prm.mat)
        return prm
    end
end

dimension(prm::PositiveSemidefiniteCone) = prm.dim
barrierpar_prm(prm::PositiveSemidefiniteCone) = prm.side
function getintdir_prm!(arr::AbstractVector{Float64}, prm::PositiveSemidefiniteCone)
    for i in 1:prm.side, j in i:prm.side
        if i == j
            prm.mat[i,j] = 1.0
        else
            prm.mat[i,j] = prm.mat[j,i] = 0.0
        end
    end
    mattovec!(arr, prm.mat)
    return arr
end
loadpnt_prm!(prm::PositiveSemidefiniteCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::PositiveSemidefiniteCone)
    vectomat!(prm.mat, prm.pnt)
    F = cholesky!(Symmetric(prm.mat), check=false)
    if !issuccess(F)
        return false
    end

    prm.matinv = -inv(F) # TODO eliminate allocs
    vectomat!(prm.matpnt, prm.pnt)
    return true
end

calcg_prm!(g::AbstractVector{Float64}, prm::PositiveSemidefiniteCone) = (mattovec!(g, prm.matinv); g)

function calcHiarr_prm!(prod::AbstractVector{Float64}, arr::AbstractVector{Float64}, prm::PositiveSemidefiniteCone)
    vectomat!(prm.mat, arr)
    mul!(prm.mat2, prm.mat, prm.matpnt)
    mul!(prm.mat, prm.matpnt, prm.mat2)
    mattovec!(prod, prm.mat)
    return prod
end

function calcHiarr_prm!(prod::AbstractMatrix{Float64}, arr::AbstractMatrix{Float64}, prm::PositiveSemidefiniteCone)
    for j in 1:size(arr, 2)
        vectomat!(prm.mat, view(arr, :, j))
        mul!(prm.mat2, prm.mat, prm.matpnt)
        mul!(prm.mat, prm.matpnt, prm.mat2)
        mattovec!(view(prod, :, j), prm.mat)
    end
    return prod
end

function calcHarr_prm!(prod::AbstractVector{Float64}, arr::AbstractVector{Float64}, prm::PositiveSemidefiniteCone)
    vectomat!(prm.mat, arr)
    mul!(prm.mat2, prm.mat, prm.matinv)
    mul!(prm.mat, prm.matinv, prm.mat2)
    mattovec!(prod, prm.mat)
    return prod
end

function calcHarr_prm!(prod::AbstractMatrix{Float64}, arr::AbstractMatrix{Float64}, prm::PositiveSemidefiniteCone)
    for j in 1:size(arr, 2)
        vectomat!(prm.mat, view(arr, :, j))
        mul!(prm.mat2, prm.mat, prm.matinv)
        mul!(prm.mat, prm.matinv, prm.mat2)
        mattovec!(view(prod, :, j), prm.mat)
    end
    return prod
end
