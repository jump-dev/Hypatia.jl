#=
Copyright 2018, Chris Coey and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(u^2 - norm(w)^2)
=#

mutable struct EpiNormEucl <: Cone
    usedual::Bool
    dim::Int
    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    Hi::Matrix{Float64}

    function EpiNormEucl(dim::Int, isdual::Bool)
        cone = new()
        cone.usedual = isdual
        cone.dim = dim
        cone.g = Vector{Float64}(undef, dim)
        cone.H = Matrix{Float64}(undef, dim, dim)
        cone.Hi = similar(cone.H)
        return cone
    end
end

EpiNormEucl(dim::Int) = EpiNormEucl(dim, false)

get_nu(cone::EpiNormEucl) = 1

set_initial_point(arr::AbstractVector{Float64}, cone::EpiNormEucl) = (@. arr = 0.0; arr[1] = 1.0; arr)

function check_in_cone(cone::EpiNormEucl)
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    if u <= 0.0
        return false
    end
    dist = abs2(u) - sum(abs2, w)
    if dist <= 0.0
        return false
    end

    @. cone.g = cone.point / dist
    cone.g[1] *= -1.0

    Hi = cone.Hi
    mul!(Hi, cone.point, cone.point') # TODO syrk
    @. Hi += Hi
    Hi[1, 1] -= dist
    for j in 2:cone.dim
        Hi[j, j] += dist
    end

    H = cone.H
    @. H = Hi
    for j in 2:cone.dim
        H[1, j] = H[j, 1] = -H[j, 1]
    end
    @. H *= abs2(inv(dist))

    return true
end

# calcg!(g::AbstractVector{Float64}, cone::EpiNormEucl) = (@. g = cone.point / dist; g[1] = -g[1]; g)
# calcHiarr!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::EpiNormEucl) = mul!(prod, cone.Hi, arr)
# calcHarr!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::EpiNormEucl) = mul!(prod, cone.H, arr)

inv_hess(cone::EpiNormEucl) = Symmetric(cone.Hi)
