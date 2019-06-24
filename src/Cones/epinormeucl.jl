#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of Euclidean (2-)norm (AKA second-order cone)
(u in R, w in R^n) : u >= norm_2(w)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-log(u^2 - norm_2(w)^2)
=#

mutable struct EpiNormEucl{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int

    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    Hi::Matrix{T}

    function EpiNormEucl{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiNormEucl{T}(dim::Int) where {T <: HypReal} = EpiNormEucl{T}(dim, false)

function setup_data(cone::EpiNormEucl{T}) where {T <: HypReal}
    dim = cone.dim
    cone.g = Vector{T}(undef, dim)
    cone.H = Matrix{T}(undef, dim, dim)
    cone.Hi = similar(cone.H)
    return
end

get_nu(cone::EpiNormEucl) = 2

set_initial_point(arr::AbstractVector{T}, cone::EpiNormEucl{T}) where {T <: HypReal} = (@. arr = zero(T); arr[1] = one(T); arr)

function check_in_cone(cone::EpiNormEucl{T}) where {T <: HypReal}
    if cone.point[1] <= zero(T)
        return false
    end
    dist = (abs2(cone.point[1]) - sum(abs2, view(cone.point, 2:cone.dim))) / 2
    if dist <= zero(T)
        return false
    end

    @. cone.g = cone.point / dist
    cone.g[1] *= -1

    # TODO only work with upper triangle
    mul!(cone.H, cone.g, cone.g')
    cone.H += inv(dist) * I
    cone.H[1, 1] -= 2 / dist

    mul!(cone.Hi, cone.point, cone.point')
    cone.Hi += dist * I
    cone.Hi[1, 1] -= 2 * dist

    return true
end

inv_hess(cone::EpiNormEucl) = Symmetric(cone.Hi, :U)

inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormEucl{T}) where {T <: HypReal} = mul!(prod, Symmetric(cone.Hi, :U), arr)
