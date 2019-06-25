#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of L-infinity norm
(u in R, w in R^n) : u >= norm_inf(w)

barrier from "Barrier Functions in Interior Point Methods" by Osman Guler
-sum_i(log(u - w_i^2/u)) - log(u)
=#

mutable struct EpiNormInf{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int

    point::AbstractVector{T}
    g::Vector{T}
    diag11::T
    diag2n::Vector{T}
    edge2n::Vector{T}
    div2n::Vector{T}
    schur::T

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
    cone.diag2n = zeros(T, dim - 1)
    cone.edge2n = zeros(T, dim - 1)
    cone.div2n = zeros(T, dim - 1)
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

    g = cone.g
    usqr = abs2(u)
    g1 = zero(T)
    h1 = zero(T)
    for j in eachindex(w)
        iuwj = T(2) / (usqr - abs2(w[j]))
        g1 += iuwj
        wiuwj = w[j] * iuwj
        h1 += abs2(iuwj)
        g[j + 1] = wiuwj

        cone.diag2n[j] = iuwj + abs2(wiuwj) # diagonal
        cone.edge2n[j] = -iuwj * wiuwj * u  # edge
        cone.div2n[j] = -cone.edge2n[j] / cone.diag2n[j] # -edge / diag
    end
    t1 = T(cone.dim - 2) / u
    g[1] = t1 - u * g1

    cone.diag11 = -t1 / u + usqr * h1 - g1
    cone.schur = cone.diag11 + dot(cone.edge2n, cone.div2n)

    return true
end

function hess(cone::EpiNormInf{T}) where {T <: HypReal}
    H = zeros(T, cone.dim, cone.dim)
    H[1, 1] = cone.diag11
    @inbounds for j in 2:cone.dim
        H[j, j] = cone.diag2n[j - 1]
        H[1, j] = cone.edge2n[j - 1]
    end
    return Symmetric(H, :U)
end

function inv_hess(cone::EpiNormInf{T}) where {T <: HypReal}
    # Hessian inverse is Diag(0, inv(diag)) + xx'/schur where x = (-1, edge ./ diag)
    Hi = zeros(T, cone.dim, cone.dim)
    Hi[1, 1] = 1
    @. Hi[1, 2:end] = cone.div2n
    @inbounds for j in 2:cone.dim, i in 2:j
        Hi[i, j] = Hi[1, j] * Hi[1, i]
    end
    Hi ./= cone.schur
    @inbounds for j in 2:cone.dim
        Hi[j, j] += inv(cone.diag2n[j - 1])
    end
    return Symmetric(Hi, :U)
end

function hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormInf{T}) where {T <: HypReal}
    @inbounds for j in 1:size(prod, 2)
        @views prod[1, j] = cone.diag11 * arr[1, j] + dot(cone.edge2n, arr[2:end, j])
        @views @. prod[2:end, j] = cone.edge2n * arr[1, j] + cone.diag2n * arr[2:end, j]
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormInf{T}) where {T <: HypReal}
    @inbounds for j in 1:size(prod, 2)
        @views prod[1, j] = arr[1, j] + dot(cone.div2n, arr[2:end, j])
        @views @. prod[2:end, j] = cone.div2n * prod[1, j]
    end
    prod ./= cone.schur
    @. @views prod[2:end, :] += arr[2:end, :] / cone.diag2n
    return prod
end
