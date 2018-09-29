
# rotated second order cone
# barrier is -ln(2*x*y - norm(z)^2)
# from Nesterov & Todd "Self-Scaled Barriers and Interior-Point Methods for Convex Programming"
mutable struct RotatedSecondOrderCone <: PrimitiveCone
    dim::Int
    pnt::AbstractVector{Float64}
    disth::Float64
    Hi::Matrix{Float64}
    H::Matrix{Float64}

    function RotatedSecondOrderCone(dim::Int)
        prm = new()
        prm.dim = dim
        prm.Hi = Matrix{Float64}(undef, dim, dim)
        prm.H = copy(prm.Hi)
        return prm
    end
end

dimension(prm::RotatedSecondOrderCone) = prm.dim
barrierpar_prm(prm::RotatedSecondOrderCone) = 2
getintdir_prm!(arr::AbstractVector{Float64}, prm::RotatedSecondOrderCone) = (@. arr[1:2] = 1.0; @. arr[3:end] = 0.0; arr)
loadpnt_prm!(prm::RotatedSecondOrderCone, pnt::AbstractVector{Float64}) = (prm.pnt = pnt)

function incone_prm(prm::RotatedSecondOrderCone)
    pnt = prm.pnt
    if (pnt[1] <= 0) || (pnt[2] <= 0)
        return false
    end
    nrm2 = 0.5*sum(abs2, pnt[j] for j in 3:prm.dim)
    prm.disth = pnt[1]*pnt[2] - nrm2
    if prm.disth <= 0.0
        return false
    end

    mul!(prm.Hi, pnt, pnt')
    prm.Hi[2,1] = prm.Hi[1,2] = nrm2
    for j in 3:prm.dim
        prm.Hi[j,j] += prm.disth
    end
    @. prm.H = prm.Hi
    for j in 3:prm.dim
        prm.H[1,j] = prm.H[j,1] = -prm.Hi[2,j]
        prm.H[2,j] = prm.H[j,2] = -prm.Hi[1,j]
    end
    prm.H[1,1] = prm.Hi[2,2]
    prm.H[2,2] = prm.Hi[1,1]
    @. prm.H *= inv(prm.disth)^2
    return true
end

calcg_prm!(g::AbstractVector{Float64}, prm::RotatedSecondOrderCone) = (@. g = prm.pnt/prm.disth; tmp = g[1]; g[1] = -g[2]; g[2] = -tmp; g)
calcHiarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::RotatedSecondOrderCone) = mul!(prod, prm.Hi, arr)
calcHarr_prm!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, prm::RotatedSecondOrderCone) = mul!(prod, prm.H, arr)
