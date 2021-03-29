#=
functions and caches for cones

TODO maybe write a fallback dual feas check that checks if ray of dual point intersects dikin ellipsoid at primal point
=#

module Cones

using LinearAlgebra
import LinearAlgebra.copytri!
import LinearAlgebra.HermOrSym
import LinearAlgebra.BlasReal
import PolynomialRoots
using SparseArrays
import SuiteSparse.CHOLMOD
import Hypatia.RealOrComplex
import Hypatia.DenseSymCache
import Hypatia.DensePosDefCache
import Hypatia.load_matrix
import Hypatia.update_fact
import Hypatia.inv_prod
import Hypatia.sqrt_prod
import Hypatia.inv_sqrt_prod
import Hypatia.invert
import Hypatia.increase_diag!
import Hypatia.outer_prod

default_use_heuristic_neighborhood() = false

# hessian_cache(T::Type{<:BlasReal}) = DenseSymCache{T}() # use Bunch Kaufman for BlasReals from start
hessian_cache(T::Type{<:Real}) = DensePosDefCache{T}()

abstract type Cone{T <: Real} end

include("nonnegative.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epiperentropy.jl")
include("epipertraceentropytri.jl")
include("epipersquare.jl")
include("epirelentropy.jl")
include("hypoperlog.jl")
include("generalizedpower.jl")
include("hypopowermean.jl")
include("hypogeomean.jl")
include("epinormspectral.jl")
include("linmatrixineq.jl")
include("possemideftri.jl")
include("possemideftrisparse.jl")
include("doublynonnegativetri.jl")
include("matrixepipersquare.jl")
include("hypoperlogdettri.jl")
include("hyporootdettri.jl")
include("epitracerelentropytri.jl")
include("wsosinterpnonnegative.jl")
include("wsosinterpepinormone.jl")
include("wsosinterpepinormeucl.jl")
include("wsosinterppossemideftri.jl")

use_dual_barrier(cone::Cone) = cone.use_dual_barrier
dimension(cone::Cone) = cone.dim

function setup_data(cone::Cone{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    if hasfield(typeof(cone), :correction)
        cone.correction = zeros(T, dim)
    end
    cone.vec1 = zeros(T, dim)
    cone.vec2 = zeros(T, dim)
    setup_extra_data(cone)
    return cone
end

load_point(cone::Cone{T}, point::AbstractVector{T}, scal::T) where {T <: Real} = (@. cone.point = scal * point)
load_point(cone::Cone, point::AbstractVector) = copyto!(cone.point, point)
load_dual_point(cone::Cone, point::AbstractVector) = copyto!(cone.dual_point, point)

is_feas(cone::Cone) = (cone.feas_updated ? cone.is_feas : update_feas(cone))
grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))
hess(cone::Cone) = (cone.hess_updated ? cone.hess : update_hess(cone))
inv_hess(cone::Cone) = (cone.inv_hess_updated ? cone.inv_hess : update_inv_hess(cone))

# fallbacks

reset_data(cone::Cone) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

function use_sqrt_oracles(cone::Cone)
    if !cone.hess_fact_updated
        update_hess_fact(cone) || return false
    end
    return (cone.hess_fact_cache isa DensePosDefCache)
end

use_correction(::Cone) = true

update_hess_aux(cone::Cone) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    if !cone.hess_updated
        update_hess(cone)
    end
    mul!(prod, cone.hess, arr)
    return prod
end

function update_hess_fact(cone::Cone{T}) where {T <: Real}
    cone.hess_fact_updated && return true
    if !cone.hess_updated
        update_hess(cone)
    end

    if !update_fact(cone.hess_fact_cache, cone.hess)
        if T <: BlasReal && cone.hess_fact_cache isa DensePosDefCache{T}
            # @warn("switching Hessian cache from Cholesky to Bunch Kaufman")
            cone.hess_fact_cache = DenseSymCache{T}()
            load_matrix(cone.hess_fact_cache, cone.hess)
            update_fact(cone.hess_fact_cache, cone.hess) || return false
        else
            return false
        end
    end

    cone.hess_fact_updated = true
    return true
end

function update_inv_hess(cone::Cone)
    update_hess_fact(cone)
    invert(cone.hess_fact_cache, cone.inv_hess)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    update_hess_fact(cone)
    copyto!(prod, arr)
    inv_prod(cone.hess_fact_cache, prod)
    return prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    update_hess_fact(cone)
    copyto!(prod, arr)
    sqrt_prod(cone.hess_fact_cache, prod)
    return prod
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    update_hess_fact(cone)
    copyto!(prod, arr)
    inv_sqrt_prod(cone.hess_fact_cache, prod)
    return prod
end

# number of nonzeros in the Hessian and inverse
hess_nz_count(cone::Cone) = dimension(cone) ^ 2
hess_nz_count_tril(cone::Cone) = svec_length(dimension(cone))
inv_hess_nz_count(cone::Cone) = dimension(cone) ^ 2
inv_hess_nz_count_tril(cone::Cone) = svec_length(dimension(cone))
# row indices of nonzero elements in column j
hess_nz_idxs_col(cone::Cone, j::Int) = 1:dimension(cone)
hess_nz_idxs_col_tril(cone::Cone, j::Int) = j:dimension(cone)
inv_hess_nz_idxs_col(cone::Cone, j::Int) = 1:dimension(cone)
inv_hess_nz_idxs_col_tril(cone::Cone, j::Int) = j:dimension(cone)

use_heuristic_neighborhood(cone::Cone) = cone.use_heuristic_neighborhood

function in_neighborhood(
    cone::Cone{T},
    rtmu::T,
    max_nbhd::T,
    ) where {T <: Real}
    is_feas(cone) || return false
    g = grad(cone)
    vec1 = cone.vec1

    # check numerics of barrier oracles
    # TODO tune
    tol = sqrt(eps(T))
    gtol = sqrt(tol)
    Htol = 10sqrt(gtol)
    dim = dimension(cone)
    nu = get_nu(cone)
    # grad check
    if abs(1 + dot(g, cone.point) / nu) > gtol * dim
        return false
    end
    # hess check
    hess_prod!(vec1, cone.point, cone)
    if abs(1 - dot(vec1, cone.point) / nu) > Htol * dim
        return false
    end
    # inv hess check
    inv_hess_prod!(vec1, g, cone)
    if abs(1 - dot(vec1, g) / nu) > Htol * dim
        return false
    end

    # check neighborhood condition
    @. vec1 = cone.dual_point + rtmu * g
    if use_heuristic_neighborhood(cone)
        nbhd = norm(vec1, Inf) / norm(g, Inf)
        # nbhd = maximum(abs(dj / gj) for (dj, gj) in zip(vec1, g)) # TODO try this neighborhood
    else
        vec2 = cone.vec2
        inv_hess_prod!(vec2, vec1, cone)
        nbhd_sqr = dot(vec2, vec1)
        if nbhd_sqr < -tol * dim
            return false
        end
        nbhd = sqrt(abs(nbhd_sqr))
    end

    return (nbhd < rtmu * max_nbhd)
end

# utilities for arrays

svec_length(side::Int) = div(side * (side + 1), 2)

svec_idx(row::Int, col::Int) = (div((row - 1) * row, 2) + col)

block_idxs(incr::Int, block::Int) = (incr * (block - 1) .+ (1:incr))

# TODO rt2::T doesn't work with tests using ForwardDiff
function smat_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{T}, rt2::Number) where T
    k = 1
    m = size(mat, 1)
    for j in 1:m, i in 1:j
        @inbounds if i == j
            vec[k] = mat[i, j]
        else
            vec[k] = mat[i, j] * rt2
        end
        k += 1
    end
    return vec
end

function smat_to_svec_add!(vec::AbstractVector{T}, mat::AbstractMatrix{T}, rt2::Number) where T
    k = 1
    m = size(mat, 1)
    for j in 1:m, i in 1:j
        @inbounds if i == j
            vec[k] += mat[i, j]
        else
            vec[k] += mat[i, j] * rt2
        end
        k += 1
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix{T}, vec::AbstractVector{T}, rt2::Number) where T
    k = 1
    m = size(mat, 1)
    for j in 1:m, i in 1:j
        @inbounds if i == j
            mat[i, j] = vec[k]
        else
            mat[i, j] = vec[k] / rt2
        end
        k += 1
    end
    return mat
end

function smat_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{Complex{T}}, rt2::Number) where T
    k = 1
    m = size(mat, 1)
    for j in 1:m, i in 1:j
        @inbounds if i == j
            vec[k] = real(mat[i, j])
            k += 1
        else
            ck = mat[i, j] * rt2
            vec[k] = real(ck)
            k += 1
            vec[k] = -imag(ck)
            k += 1
        end
    end
    return vec
end

function smat_to_svec_add!(vec::AbstractVector{T}, mat::AbstractMatrix{Complex{T}}, rt2::Number) where T
    k = 1
    m = size(mat, 1)
    for j in 1:m, i in 1:j
        @inbounds if i == j
            vec[k] += real(mat[i, j])
            k += 1
        else
            ck = mat[i, j] * rt2
            vec[k] += real(ck)
            k += 1
            vec[k] -= imag(ck)
            k += 1
        end
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix{Complex{T}}, vec::AbstractVector{T}, rt2::Number) where T
    k = 1
    m = size(mat, 1)
    @inbounds for j in 1:m, i in 1:j
        if i == j
            mat[i, j] = vec[k]
            k += 1
        else
            mat[i, j] = Complex(vec[k], -vec[k + 1]) / rt2
            k += 2
        end
    end
    return mat
end

# utilities for converting between real and complex vectors
function rvec_to_cvec!(cvec::AbstractVecOrMat{Complex{T}}, rvec::AbstractVecOrMat{T}) where T
    k = 1
    @inbounds for i in eachindex(cvec)
        cvec[i] = Complex(rvec[k], rvec[k + 1])
        k += 2
    end
    return cvec
end

function cvec_to_rvec!(rvec::AbstractVecOrMat{T}, cvec::AbstractVecOrMat{Complex{T}}) where T
    k = 1
    @inbounds for i in eachindex(cvec)
        ci = cvec[i]
        rvec[k] = real(ci)
        rvec[k + 1] = imag(ci)
        k += 2
    end
    return rvec
end

vec_copy_to!(v1::AbstractVecOrMat{T}, v2::AbstractVecOrMat{T}) where {T <: Real} = copyto!(v1, v2)
vec_copy_to!(v1::AbstractVecOrMat{T}, v2::AbstractVecOrMat{Complex{T}}) where {T <: Real} = cvec_to_rvec!(v1, v2)
vec_copy_to!(v1::AbstractVecOrMat{Complex{T}}, v2::AbstractVecOrMat{T}) where {T <: Real} = rvec_to_cvec!(v1, v2)

# utilities for hessians for cones with PSD parts

# TODO parallelize
function symm_kron(H::AbstractMatrix{T}, mat::AbstractMatrix{T}, rt2::T; upper_only::Bool = true) where {T <: Real}
    side = size(mat, 1)
    col_idx = 1
    @inbounds for l in 1:side
        for k in 1:(l - 1)
            row_idx = 1
            for j in 1:side
                upper_only && row_idx > col_idx && continue
                for i in 1:(j - 1)
                    scal = (i == j ? 1 : rt2) * (k == l ? 1 : rt2) / 2
                    H[row_idx, col_idx] = scal * (mat[i, k] * mat[j, l] + mat[i, l] * mat[j, k])
                    row_idx += 1
                end
                H[row_idx, col_idx] = rt2 * mat[j, k] * mat[j, l]
                row_idx += 1
            end
            col_idx += 1
        end
        row_idx = 1
        for j in 1:side
            upper_only && row_idx > col_idx && continue
            for i in 1:(j - 1)
                H[row_idx, col_idx] = rt2 * mat[i, l] * mat[j, l]
                row_idx += 1
            end
            H[row_idx, col_idx] = abs2(mat[j, l])
            row_idx += 1
        end
        col_idx += 1
    end
    return H
end

# TODO test output for non-Hermitian mat, the result may need transposing
function symm_kron(H::AbstractMatrix{T}, mat::AbstractMatrix{Complex{T}}, rt2::T) where {T <: Real}
    side = size(mat, 1)
    col_idx = 1
    for i in 1:side, j in 1:i
        row_idx = 1
        if i == j
            @inbounds for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    H[row_idx, col_idx] = abs2(mat[i2, i])
                    row_idx += 1
                else
                    c = rt2 * mat[i, i2] * mat[j2, j]
                    H[row_idx, col_idx] = real(c)
                    row_idx += 1
                    H[row_idx, col_idx] = -imag(c)
                    row_idx += 1
                end
                if row_idx > col_idx
                    break
                end
            end
            col_idx += 1
        else
            @inbounds for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    c = rt2 * mat[i2, i] * mat[j, j2]
                    H[row_idx, col_idx] = real(c)
                    H[row_idx, col_idx + 1] = -imag(c)
                    row_idx += 1
                else
                    b1 = mat[i2, i] * mat[j, j2]
                    b2 = mat[j2, i] * mat[j, i2]
                    c1 = b1 + b2
                    H[row_idx, col_idx] = real(c1)
                    H[row_idx, col_idx + 1] = -imag(c1)
                    row_idx += 1
                    c2 = b1 - b2
                    H[row_idx, col_idx] = imag(c2)
                    H[row_idx, col_idx + 1] = real(c2)
                    row_idx += 1
                end
                if row_idx > col_idx
                    break
                end
            end
            col_idx += 2
        end
    end
    return H
end

function hess_element(H::Matrix{T}, r_idx::Int, c_idx::Int, term1::T, term2::T) where {T <: Real}
    @inbounds H[r_idx, c_idx] = term1 + term2
    return
end

function hess_element(H::Matrix{T}, r_idx::Int, c_idx::Int, term1::Complex{T}, term2::Complex{T}) where {T <: Real}
    @inbounds begin
        H[r_idx, c_idx] = real(term1) + real(term2)
        H[r_idx + 1, c_idx] = imag(term2) - imag(term1)
        H[r_idx, c_idx + 1] = imag(term1) + imag(term2)
        H[r_idx + 1, c_idx + 1] = real(term1) - real(term2)
    end
    return
end

# set up sparse arrow matrix data structure
# TODO remove this in favor of new hess_nz_count etc functions that directly use uu, uw, ww etc
function sparse_arrow(T::Type{<:Real}, w_dim::Int)
    dim = w_dim + 1
    nnz_tri = 2 * dim - 1
    I = Vector{Int}(undef, nnz_tri)
    J = Vector{Int}(undef, nnz_tri)
    idxs1 = 1:dim
    @views I[idxs1] .= 1
    @views J[idxs1] .= idxs1
    idxs2 = (dim + 1):(2 * dim - 1)
    @views I[idxs2] .= 2:dim
    @views J[idxs2] .= 2:dim
    V = ones(T, nnz_tri)
    return sparse(I, J, V, dim, dim)
end

# 2x2 block case
function sparse_arrow_block2(T::Type{<:Real}, w_dim::Int)
    dim = 2 * w_dim + 1
    nnz_tri = 2 * dim - 1 + w_dim
    I = Vector{Int}(undef, nnz_tri)
    J = Vector{Int}(undef, nnz_tri)
    idxs1 = 1:dim
    @views I[idxs1] .= 1
    @views J[idxs1] .= idxs1
    idxs2 = (dim + 1):(2 * dim - 1)
    @views I[idxs2] .= 2:dim
    @views J[idxs2] .= 2:dim
    idxs3 = (2 * dim):nnz_tri
    @views I[idxs3] .= 2:2:dim
    @views J[idxs3] .= 3:2:dim
    V = ones(T, nnz_tri)
    return sparse(I, J, V, dim, dim)
end

# lmul with arrow matrix
function arrow_prod(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, uu::T, uw::Vector{T}, ww::Vector{T}) where {T <: Real}
    @inbounds @views begin
        arru = arr[1, :]
        arrw = arr[2:end, :]
        produ = prod[1, :]
        prodw = prod[2:end, :]
        copyto!(produ, arru)
        mul!(produ, arrw', uw, true, uu)
        mul!(prodw, uw, arru')
        @. prodw += ww * arrw
    end
    return prod
end

# 2x2 block case
function arrow_prod(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, uu::T, uv::Vector{T}, uw::Vector{T}, vv::Vector{T}, vw::Vector{T}, ww::Vector{T}) where {T <: Real}
    @inbounds @views begin
        arru = arr[1, :]
        arrv = arr[2:2:end, :]
        arrw = arr[3:2:end, :]
        produ = prod[1, :]
        prodv = prod[2:2:end, :]
        prodw = prod[3:2:end, :]
        @. produ = uu * arru
        mul!(produ, arrv', uv, true, true)
        mul!(produ, arrw', uw, true, true)
        mul!(prodv, uv, arru')
        mul!(prodw, uw, arru')
        @. prodv += vv * arrv + vw * arrw
        @. prodw += ww * arrw + vw * arrv
    end
    return prod
end

# factorize arrow matrix
function arrow_sqrt(uu::T, uw::Vector{T}, ww::Vector{T}, rtuw::Vector{T}, rtww::Vector{T}) where {T <: Real}
    tol = sqrt(eps(T))
    any(<(tol), ww) && return zero(T)
    @. rtww = sqrt(ww)
    @. rtuw = uw / rtww
    diff = uu - sum(abs2, rtuw)
    (diff < tol) && return zero(T)
    return sqrt(diff)
end

# 2x2 block case
function arrow_sqrt(uu::T, uv::Vector{T}, uw::Vector{T}, vv::Vector{T}, vw::Vector{T}, ww::Vector{T}, rtuv::Vector{T}, rtuw::Vector{T}, rtvv::Vector{T}, rtvw::Vector{T}, rtww::Vector{T}) where {T <: Real}
    tol = sqrt(eps(T))
    any(<(tol), ww) && return zero(T)
    @. rtww = sqrt(ww)
    @. rtvw = vw / rtww
    @. rtuw = uw / rtww
    @. rtuv = vv - abs2(rtvw)
    any(<(tol), rtuv) && return zero(T)
    @. rtvv = sqrt(rtuv)
    @. rtuv = (uv - rtvw * rtuw) / rtvv
    diff = uu - sum(abs2, rtuv) - sum(abs2, rtuw)
    (diff < tol) && return zero(T)
    return sqrt(diff)
end

# lmul with lower Cholesky factor of arrow matrix
function arrow_sqrt_prod(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, rtuu::T, rtuw::Vector{T}, rtww::Vector{T}) where {T <: Real}
    @inbounds @views begin
        arr1 = arr[1, :]
        @. prod[1, :] = rtuu * arr1
        @. prod[2:end, :] = rtuw * arr1' + rtww * arr[2:end, :]
    end
    return prod
end

# 2x2 block case
function arrow_sqrt_prod(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, rtuu::T, rtuv::Vector{T}, rtuw::Vector{T}, rtvv::Vector{T}, rtvw::Vector{T}, rtww::Vector{T}) where {T <: Real}
    @inbounds @views begin
        arr1 = arr[1, :]
        arrv = arr[2:2:end, :]
        @. prod[1, :] = rtuu * arr1
        @. prod[2:2:end, :] = rtuv * arr1' + rtvv * arrv
        @. prod[3:2:end, :] = rtuw * arr1' + rtvw * arrv + rtww * arr[3:2:end, :]
    end
    return prod
end

# ldiv with upper Cholesky factor of arrow matrix
function inv_arrow_sqrt_prod(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, rtuu::T, rtuw::Vector{T}, rtww::Vector{T}) where {T <: Real}
    @inbounds @. @views prod[2:end, :] = arr[2:end, :] / rtww
    @inbounds @views for j in 1:size(arr, 2)
        prod[1, j] = (arr[1, j] - dot(prod[2:end, j], rtuw)) / rtuu
    end
    return prod
end

# 2x2 block case
function inv_arrow_sqrt_prod(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, rtuu::T, rtuv::Vector{T}, rtuw::Vector{T}, rtvv::Vector{T}, rtvw::Vector{T}, rtww::Vector{T}) where {T <: Real}
    @inbounds @views begin
        prodw = prod[3:2:end, :]
        @. prodw = arr[3:2:end, :] / rtww
        @. prod[2:2:end, :] = (arr[2:2:end, :] - rtvw * prodw) / rtvv
    end
    @inbounds @views for j in 1:size(arr, 2)
        prod[1, j] = (arr[1, j] - dot(prod[2:2:end, j], rtuv) - dot(prod[3:2:end, j], rtuw)) / rtuu
    end
    return prod
end

function grad_logm!(
        mat::Matrix{T},
        vecs::Matrix{T},
        tempmat1::Matrix{T},
        tempmat2::Matrix{T},
        tempvec::Vector{T},
        diff_mat::AbstractMatrix{T},
        rt2::T,
        ) where T
    veckron = symm_kron(tempmat1, vecs, rt2, upper_only = false)
    smat_to_svec!(tempvec, diff_mat, one(T))
    mul!(tempmat2, veckron, Diagonal(tempvec))
    return mul!(mat, tempmat2, veckron')
end

function diff_mat!(mat::Matrix{T}, vals::Vector{T}, log_vals::Vector{T}) where T
    rteps = sqrt(eps(T))
    @inbounds for j in eachindex(vals)
        (vj, lvj) = (vals[j], log_vals[j])
        for i in 1:(j - 1)
            (vi, lvi) = (vals[i], log_vals[i])
            mat[i, j] = (abs(vi - vj) < rteps ? inv((vi + vj) / 2) : (lvi - lvj) / (vi - vj))
        end
        mat[j, j] = inv(vj)
    end
    return mat
end

function diff_tensor!(diff_tensor::Array{T, 3}, diff_mat::AbstractMatrix{T}, vals::Vector{T}) where T
    rteps = sqrt(eps(T))
    d = size(diff_mat, 1)
    @inbounds for k in 1:d, j in 1:k, i in 1:j
        (vi, vj, vk) = (vals[i], vals[j], vals[k])
        if abs(vj - vk) < rteps
            if abs(vi - vj) < rteps
                vijk = (vi + vj + vk) / 3
                t = -inv(vijk) / vijk / 2
            else
                # diff_mat[j, k] ≈ diff_mat[j, j]
                t = (diff_mat[j, k] - diff_mat[i, j]) / (vj - vi)
            end
        else
            t = (diff_mat[i, j] - diff_mat[i, k]) / (vj - vk)
        end
        diff_tensor[i, j, k] = diff_tensor[i, k, j] = diff_tensor[j, i, k] =
            diff_tensor[j, k, i] = diff_tensor[k, i, j] = diff_tensor[k, j, i] = t
    end
    return diff_tensor
end

end
