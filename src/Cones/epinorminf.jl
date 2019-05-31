#=
Copyright 2018, Chris Coey and contributors

epigraph of L-infinity norm
(u in R, w in R^n) : u >= norm_inf(w)

barrier from "Barrier Functions in Interior Point Methods" by Osman Guler
-sum_i(log(u^2 - w_i^2)) + (n-1)*log(u)

TODO for efficiency, don't construct full H matrix (arrow fill)
=#

mutable struct EpiNormInf{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int

    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    F

    function EpiNormInf{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

EpiNormInf{T}(dim::Int) where {T <: HypReal} = EpiNormInf{T}(dim, false)

function setup_data(cone::EpiNormInf{T}) where {T <: HypReal}
    dim = cone.dim
    cone.g = zeros(T, dim)
    cone.H = zeros(T, dim, dim)
    cone.H2 = copy(cone.H)
    return
end

get_nu(cone::EpiNormInf) = cone.dim

set_initial_point(arr::AbstractVector{T}, cone::EpiNormInf{T}) where {T <: HypReal} = (@. arr = zero(T); arr[1] = one(T); arr)

function check_in_cone(cone::EpiNormInf{T}) where {T <: HypReal}
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    if u <= maximum(abs, w)
        return false
    end

    # TODO don't explicitly construct full matrix (arrow)
    g = cone.g
    H = cone.H
    usqr = abs2(u)
    g1 = zero(T)
    h1 = zero(T)
    for j in eachindex(w)
        iuwj = T(2) / (usqr - abs2(w[j]))
        g1 += iuwj
        wiuwj = w[j] * iuwj
        h1 += abs2(iuwj)
        jp1 = j + 1
        g[jp1] = wiuwj
        H[jp1, jp1] = iuwj + abs2(wiuwj)
        H[1, jp1] = H[jp1, 1] = -iuwj * wiuwj * u
    end
    invu = inv(u)
    t1 = (cone.dim - 2) * invu
    g[1] = t1 - u * g1
    H[1, 1] = -t1 * invu + usqr * h1 - g1

    return factorize_hess(cone)
end
