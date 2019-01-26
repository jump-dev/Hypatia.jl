#=
Copyright 2018, Chris Coey and contributors

epigraph of L-infinity norm
(u in R, w in R^n) : u >= norm_inf(w)

barrier from "Barrier Functions in Interior Point Methods" by Osman Guler
-sum_i(log(u^2 - w_i^2)) + (n-1)*log(u)

TODO for efficiency, don't construct full H matrix (arrow fill)
=#

mutable struct EpiNormInf <: Cone
    usedual::Bool
    dim::Int
    primals::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F

    function EpiNormInf(dim::Int, isdual::Bool)
        cone = new()
        cone.usedual = isdual
        cone.dim = dim
        cone.g = Vector{Float64}(undef, dim)
        cone.H = Matrix{Float64}(undef, dim, dim)
        cone.H2 = similar(cone.H)
        return cone
    end
end

EpiNormInf(dim::Int) = EpiNormInf(dim, false)

get_nu(cone::EpiNormInf) = cone.dim

set_initial_point(arr::AbstractVector{Float64}, cone::EpiNormInf) = (@. arr = 0.0; arr[1] = 1.0; arr)

function check_in_cone(cone::EpiNormInf)
    u = cone.primals[1]
    w = view(cone.primals, 2:cone.dim)
    if u <= maximum(abs, w)
        return false
    end

    # TODO don't explicitly construct full matrix
    g = cone.g
    H = cone.H
    usqr = abs2(u)
    g1 = 0.0
    h1 = 0.0
    for j in eachindex(w)
        iuwj = 2.0 / (usqr - abs2(w[j]))
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
