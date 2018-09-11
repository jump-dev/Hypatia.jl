
# positive semidefinite cone (lower triangle, off-diagonals scaled)
mutable struct PositiveSemidefiniteCone <: PrimitiveCone
    dim::Int
    side::Int
    pnt::AbstractVector{Float64}
    g::Vector{Float64}
    mat::Matrix{Float64}
    mat2::Matrix{Float64}
    matpnt::Matrix{Float64}

    function PositiveSemidefiniteCone(dim::Int)
        prm = new()
        prm.dim = dim
        prm.side = round(Int, sqrt(0.25 + dim + dim) - 0.5)
        prm.g = Vector{Float64}(undef, dim)
        prm.mat = Matrix{Float64}(undef, prm.side, prm.side)
        prm.mat2 = copy(prm.mat)
        prm.matpnt = copy(prm.mat)
        return prm
    end
end

dimension(prm::PositiveSemidefiniteCone) = prm.dim
barrierpar_prm(prm::PositiveSemidefiniteCone) = prm.side
getintdir_prm!(arr::AbstractVector{Float64}, prm::PositiveSemidefiniteCone) = mattovec!(arr, Matrix(1.0I, prm.side, prm.side))
loadpnt_prm!(prm::PositiveSemidefiniteCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::PositiveSemidefiniteCone)
    vectomat!(prm.mat, prm.pnt)
    F = cholesky!(Symmetric(prm.mat), check=false)
    if !issuccess(F)
        return false
    end

    grad = -inv(F) # TODO reduce allocs
    mattovec!(prm.g, grad)
    vectomat!(prm.matpnt, prm.pnt)
    return true
end

calcg_prm!(g::AbstractVector{Float64}, prm::PositiveSemidefiniteCone) = (g .= prm.g; g)

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
