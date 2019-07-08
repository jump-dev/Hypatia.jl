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

    is_feas::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

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

# TODO maybe only allocate the fields we use
function setup_data(cone::EpiNormInf{T}) where {T <: HypReal}
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.diag2n = zeros(T, dim - 1)
    cone.edge2n = zeros(T, dim - 1)
    cone.div2n = zeros(T, dim - 1)
    return
end

get_nu(cone::EpiNormInf) = cone.dim

function set_initial_point(arr::AbstractVector, cone::EpiNormInf)
    arr .= 0
    arr[1] = 1
    return arr
end

reset_data(cone::EpiNormInf) = (cone.is_feas = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function update_feas(cone::EpiNormInf)
    @assert !cone.is_feas
    if cone.point[1] > 0
        cone.is_feas = (cone.point[1] > maximum(abs, view(cone.point, 2:cone.dim)))
    end
    return cone.is_feas
end

function update_grad(cone::EpiNormInf{T}) where {T <: HypReal}
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    g1 = zero(u)
    h1 = zero(u)
    usqr = abs2(u)
    for (j, wj) in enumerate(w)
        iuw2u = 2 / (u - abs2(wj) / u)
        g1 += iuw2u
        h1 += abs2(iuw2u)
        iu2ww = 2 / (usqr / wj - wj)
        cone.grad[j + 1] = iu2ww
        iu2w2 = 2 / (usqr - abs2(wj))
        cone.diag2n[j] = iu2w2 + abs2(iu2ww)
        cone.edge2n[j] = -iu2w2 * iuw2u
        cone.div2n[j] = iuw2u / (1 + abs2(iu2ww) / iu2w2)
    end
    t1 = (cone.dim - 2) / u
    cone.grad[1] = t1 - g1
    cone.diag11 = -(t1 + g1) / u + h1
    cone.schur = cone.diag11 + sum(cone.edge2n, cone.div2n)
    cone.grad_updated = true
    return cone.grad
end



# u = cone.point[1]
# w = view(cone.point, 2:cone.dim)
#
# usqr = abs2(u)
# g1 = zero(T)
# h1 = zero(T)
# for j in eachindex(w)
#     iuwj = 2 / (usqr - abs2(w[j]))
#     g1 += iuwj
#     h1 += abs2(iuwj)
#     wiuwj = w[j] * iuwj
#     cone.grad[j + 1] = wiuwj
#
#     cone.diag2n[j] = iuwj + abs2(wiuwj) # diagonal
#     cone.edge2n[j] = -iuwj * wiuwj * u # edge
#     cone.div2n[j] = -cone.edge2n[j] / cone.diag2n[j] # -edge / diag
# end
# t1 = (cone.dim - 2) / u
# cone.grad[1] = t1 - u * g1
#
# cone.diag11 = -t1 / u + usqr * h1 - g1
# cone.schur = cone.diag11 + dot(cone.edge2n, cone.div2n)



# function update_hess_helpers(cone::EpiNormInf)
#     @assert cone.grad_updated
#     u = cone.point[1]
#     w = view(cone.point, 2:cone.dim)
#     wg = view(cone.grad, 2:cone.dim)
#     @. cone.diag2n = abs2(wg) + wg / w
#     @. cone.edge2n = -abs2(wg) / w * u
#
#     cone.diag11 = (2 - cone.dim) / u / u +
#
#     u = cone.point[1]
#     w = view(cone.point, 2:cone.dim)
#     usqr = abs2(u)
#     g1 = zero(u)
#     for (j, wj) in enumerate(w)
#         g1 += 2 / (u - abs2(wj) / u)
#         cone.grad[j + 1] = 2 / (usqr / wj - wj)
#     end
#
#
#     cone.hess_helpers_updated = true
#     return
# end

# symmetric arrow matrix
function update_hess(cone::EpiNormInf)
    @assert cone.grad_updated
    cone.hess[1, 1] = cone.diag11
    for j in 2:cone.dim
        cone.hess[j, j] = cone.diag2n[j - 1]
        cone.hess[1, j] = cone.edge2n[j - 1]
    end
    return cone.hess
end

# Diag(0, inv(diag)) + xx' / schur, where x = (-1, edge ./ diag)
function update_inv_hess(cone::EpiNormInf)
    @assert cone.grad_updated
    cone.inv_hess[1, 1] = 1
    @. cone.inv_hess[1, 2:end] = cone.div2n
    for j in 2:cone.dim, i in 2:j
        cone.inv_hess[i, j] = cone.inv_hess[1, j] * cone.inv_hess[1, i]
    end
    cone.inv_hess ./= cone.schur
    for j in 2:cone.dim
        cone.inv_hess[j, j] += inv(cone.diag2n[j - 1])
    end
    return cone.inv_hess
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    @assert cone.grad_updated
    for j in 1:size(prod, 2)
        @views prod[1, j] = cone.diag11 * arr[1, j] + dot(cone.edge2n, arr[2:end, j])
        @views @. prod[2:end, j] = cone.edge2n * arr[1, j] + cone.diag2n * arr[2:end, j]
    end
    return prod
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormInf)
    @assert cone.grad_updated
    for j in 1:size(prod, 2)
        @views prod[1, j] = arr[1, j] + dot(cone.div2n, arr[2:end, j])
        @views @. prod[2:end, j] = cone.div2n * prod[1, j]
    end
    prod ./= cone.schur
    @. @views prod[2:end, :] += arr[2:end, :] / cone.diag2n
    return prod
end
