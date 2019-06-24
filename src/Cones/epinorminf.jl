#=
Copyright 2018, Chris Coey and contributors

epigraph of L-infinity norm
(u in R, w in R^n) : u >= norm_inf(w)

barrier from "Barrier Functions in Interior Point Methods" by Osman Guler
-sum_i(log(u^2 - w_i^2)) + (n-1)*log(u)

=#

mutable struct Aredge{T <: HypReal}
    point::T
    edge::Vector{T}
    diag::Vector{T}
end

mutable struct EpiNormInf{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int

    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    H2::Matrix{T}
    Harw::Aredge
    diagidxs::Vector{Int}
    Hi::Matrix{T}
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

    cone.Harw = Aredge(zero(T), zeros(T, dim - 1), zeros(T, dim - 1))
    cone.diagidxs = diagind(zeros(dim, dim))[2:end] # TODO calculate
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

    Harw = cone.Harw

    # TODO don't explicitly construct full matrix (aredge)
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

        Harw.diag[j] = iuwj + abs2(wiuwj) # diagonal
        Harw.edge[j] = -iuwj * wiuwj * u  # edge
    end
    invu = inv(u)
    t1 = (cone.dim - 2) * invu
    g[1] = t1 - u * g1

    H[1, 1] = -t1 * invu + usqr * h1 - g1
    Harw.point = -t1 * invu + usqr * h1 - g1

    # diag_inv =
    # diag_inv_edge = Harw[2][2:end] ./ Harw[1]
    # schur = H[1, 1] - dot(Harw[2][2:end], diag_inv_edge)
    # Hi[2:end, 2:end] = diag_inv_edge * diag_inv_edge' / schur
    # Hi[1, 1]

    return factorize_hess(cone)
end

function hess(cone::EpiNormInf)
    ret = zeros(cone.dim, cone.dim)
    ret[1, 1] = cone.Harw.point
    view(ret, 1, 2:cone.dim) .= cone.Harw.edge
    view(ret, cone.diagidxs) .= cone.Harw.diag # @. view(a, diagind(a))
    return Symmetric(ret)
end

function inv_hess(cone::EpiNormInf)
    ret = zeros(cone.dim, cone.dim)
    view(ret, cone.diagidxs) .= inv.(cone.Harw.diag)
    diag_inv_edge = H.diag ./ H.edge
    schur = H.point - dot(H.diag, diag_inv_edge)
    ret[2:end, 2:end] = diag_inv_edge * diag_inv_edge' / schur
    ret[1, 2:end] = -diag_inv_edge / schur
    @assert Symmetric(ret) * cone.H ≈ I atol=1e-5 rtol=1e-5
    return Symmetric(ret)
end

function hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormInf{T}) where {T <: HypReal}
    H = cone.Harw
    prod[1, :] = H.point * arr[1, :]' + H.edge' * arr[2:end, :]
    prod[2:end, :] = arr[2:end, :] .* H.diag + H.edge * arr[1, :]'
    return prod
end

# function inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::EpiNormInf{T}) where {T <: HypReal}
#     H = cone.Harw
#     diag_inv_edge = H.diag ./ H.edge
#     schur = H.point - dot(H.diag, diag_inv_edge)
#     x_arr = diag_inv_edge' * arr[2:end, :] - arr[1, :]
#     prod[2:cone.dim, :] = diag_inv_edge * x_arr
#     prod[1, :] = -x_arr
#     prod ./= schur
#     prod[2:end, :] .+= arr[2:end, :] ./ H.diag
#
#     # prod = diag_inv_edge * x_arr / schur
#     # prod[2:end, 2:end] += inv.(H.diag)
#     return prod
# end
