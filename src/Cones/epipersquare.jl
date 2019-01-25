#=
Copyright 2018, Chris Coey and contributors

epigraph of perspective of (half) square function (AKA rotated second-order cone)
(u in R, v in R_+, w in R^n) : u >= v*1/2*norm_2(w/v)^2
note v*1/2*norm_2(w/v)^2 = 1/2*sum_i(w_i^2)/v

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(2*u*v - norm_2(w)^2)
=#

mutable struct EpiPerSquare <: Cone
    usedual::Bool
    dim::Int
    pnt::AbstractVector{Float64}
    dist::Float64
    Hi::Matrix{Float64}
    H::Matrix{Float64}

    function EpiPerSquare(dim::Int, isdual::Bool)
        cone = new()
        cone.usedual = isdual
        cone.dim = dim
        cone.Hi = Matrix{Float64}(undef, dim, dim)
        cone.H = similar(cone.Hi)
        return cone
    end
end

EpiPerSquare(dim::Int) = EpiPerSquare(dim, false)

dimension(cone::EpiPerSquare) = cone.dim
get_nu(cone::EpiPerSquare) = 2
set_initial_point(arr::AbstractVector{Float64}, cone::EpiPerSquare) = (@. arr = 0.0; arr[1] = 1.0; arr[2] = 1.0; arr)
loadpnt!(cone::EpiPerSquare, pnt::AbstractVector{Float64}) = (cone.pnt = pnt)

function incone(cone::EpiPerSquare, scal::Float64)
    u = cone.pnt[1]
    v = cone.pnt[2]
    w = view(cone.pnt, 3:cone.dim)
    if u <= 0.0 || v <= 0.0
        return false
    end
    nrm2 = 0.5*sum(abs2, w)
    dist = u*v - nrm2
    if dist <= 0.0
        return false
    end
    cone.dist = dist

    H = cone.H
    Hi = cone.Hi
    mul!(Hi, cone.pnt, cone.pnt')
    Hi[2,1] = Hi[1,2] = nrm2
    for j in 3:cone.dim
        Hi[j,j] += dist
    end
    @. H = Hi
    for j in 3:cone.dim
        H[1,j] = H[j,1] = -Hi[2,j]
        H[2,j] = H[j,2] = -Hi[1,j]
    end
    H[1,1] = Hi[2,2]
    H[2,2] = Hi[1,1]
    @. H *= abs2(inv(dist))
    return true
end

calcg!(g::AbstractVector{Float64}, cone::EpiPerSquare) = (@. g = cone.pnt/cone.dist; tmp = g[1]; g[1] = -g[2]; g[2] = -tmp; g)
calcHiarr!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::EpiPerSquare) = mul!(prod, cone.Hi, arr)
calcHarr!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::EpiPerSquare) = mul!(prod, cone.H, arr)
