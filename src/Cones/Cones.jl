#=
Copyright 2018, Chris Coey and contributors

functions and caches for cones
=#

module Cones

using TimerOutputs
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

default_max_neighborhood() = 0.7
default_use_heuristic_neighborhood() = false

# hessian_cache(T::Type{<:BlasReal}) = DenseSymCache{T}() # use Bunch Kaufman for BlasReals from start
hessian_cache(T::Type{<:Real}) = DensePosDefCache{T}()

abstract type Cone{T <: Real} end

include("nonnegative.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epipersquare.jl")
include("power.jl")
include("hypoperlog.jl")
include("episumperentropy.jl")
include("hypogeomean.jl")
include("epinormspectral.jl")
include("matrixepipersquare.jl")
include("linmatrixineq.jl")
include("possemideftri.jl")
include("possemideftrisparse.jl")
include("doublynonnegative.jl")
include("hypoperlogdettri.jl")
include("hyporootdettri.jl")
include("wsosinterpnonnegative.jl")
include("wsosinterppossemideftri.jl")
include("wsosinterpepinormeucl.jl")

use_dual_barrier(cone::Cone) = cone.use_dual_barrier
dimension(cone::Cone) = cone.dim
set_timer(cone::Cone, timer::TimerOutput) = (cone.timer = timer)

load_point(cone::Cone{T}, point::AbstractVector{T}, scal::T) where {T <: Real} = (@. cone.point = scal * point)
load_point(cone::Cone, point::AbstractVector) = copyto!(cone.point, point)
load_dual_point(cone::Cone, point::AbstractVector) = copyto!(cone.dual_point, point)

is_feas(cone::Cone) = (cone.feas_updated ? cone.is_feas : update_feas(cone))
# is_dual_feas(cone::Cone) = # TODO maybe write a fallback dual feas check that checks if ray of dual point intersects dikin ellipsoid at primal point
grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))
hess(cone::Cone) = (cone.hess_updated ? cone.hess : update_hess(cone))
inv_hess(cone::Cone) = (cone.inv_hess_updated ? cone.inv_hess : update_inv_hess(cone))

# fallbacks

reset_data(cone::Cone) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

function use_sqrt_oracles(cone::Cone)
    cone.hess_fact_updated || update_hess_fact(cone; recover = true)
    return (cone.hess_fact_cache isa DensePosDefCache)
end

use_correction(::Cone) = true

update_hess_prod(cone::Cone) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    if !cone.hess_updated
        update_hess(cone)
    end
    mul!(prod, cone.hess, arr)
    return prod
end

function update_hess_fact(cone::Cone{T}; recover::Bool = true) where {T <: Real}
    cone.hess_fact_updated && return true
    if !cone.hess_updated
        update_hess(cone)
    end

    fact_success = update_fact(cone.hess_fact_cache, cone.hess)

    if !fact_success
        recover || return false
        # TODO if Chol, try adding sqrt(eps(T)) to diag and re-factorize
        if T <: BlasReal && cone.hess_fact_cache isa DensePosDefCache{T}
            # @warn("switching Hessian cache from Cholesky to Bunch Kaufman")
            cone.hess_fact_cache = DenseSymCache{T}()
            load_matrix(cone.hess_fact_cache, cone.hess)
        else
            # attempt recovery
            # TODO probably safer to only change the copy of the hessian that is getting factorized, not the hessian itself
            rteps = sqrt(eps(T))
            @inbounds for i in 1:size(cone.hess, 1)
                cone.hess[i, i] += rteps
            end
        end
        if !update_fact(cone.hess_fact_cache, cone.hess)
            @warn("Hessian Bunch-Kaufman factorization failed after recovery")
            return false
        end
    end

    cone.hess_fact_updated = true
    return true
end

function update_inv_hess(cone::Cone)
    @assert !cone.inv_hess_updated
    if !cone.hess_fact_updated
        update_hess_fact(cone)
    end
    invert(cone.hess_fact_cache, cone.inv_hess)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    if !cone.hess_fact_updated
        update_hess_fact(cone)
    end
    copyto!(prod, arr)
    inv_prod(cone.hess_fact_cache, prod)
    return prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    if !cone.hess_fact_updated
        update_hess_fact(cone)
    end
    copyto!(prod, arr)
    sqrt_prod(cone.hess_fact_cache, prod)
    return prod
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    if !cone.hess_fact_updated
        update_hess_fact(cone)
    end
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

function in_neighborhood(cone::Cone{T}, rtmu::T, max_nbhd::T) where {T <: Real}
    # norm(H^(-1/2) * (z + mu * grad))
    nbhd_tmp = cone.nbhd_tmp
    nbhd_tmp2 = cone.nbhd_tmp2
    g = grad(cone)
    @. nbhd_tmp = cone.dual_point + rtmu * g

    if use_heuristic_neighborhood(cone)
        nbhd = norm(nbhd_tmp, Inf) / norm(g, Inf)
        # nbhd = maximum(abs(dj / gj) for (dj, gj) in zip(nbhd_tmp, g)) # TODO try this neighborhood
    elseif Cones.use_sqrt_oracles(cone)
        inv_hess_sqrt_prod!(nbhd_tmp2, nbhd_tmp, cone)
        nbhd = norm(nbhd_tmp2)
    else
        inv_hess_prod!(nbhd_tmp2, nbhd_tmp, cone)
        nbhd_sqr = dot(nbhd_tmp2, nbhd_tmp)
        if nbhd_sqr < -eps(T) # TODO possibly loosen
            # @warn("numerical failure: cone neighborhood is $nbhd_sqr")
            return false
        end
        nbhd = sqrt(abs(nbhd_sqr))
    end

    return (nbhd < rtmu * max_nbhd)
end

# TODO cleanup / remove if not using
# function in_neighborhood(cone::Cone{T}, dual_point::AbstractVector{T}, rtmu::T, max_nbhd::T) where {T <: Real}
#     # norm(H^(-1/2) * (z + mu * grad))
#     nbhd_tmp = cone.nbhd_tmp
#     g = rtmu * grad(cone)
#     dp = copy(dual_point)
#
#     # TODO trying to see if ray intersects dikin ellipsoid
#     # TODO find point on ray closest to g in the hessian norm
#     gdp = dot(dp, g)
#     # gdp = dot(dp, hess_prod!(similar(g), g, cone)) / mu
#     # gdp = -dot(dp, cone.point)
#     if gdp > 0
#         return false
#     end
#
#     # TODO this can be used to check feasibility rather than nbhd
#     # scal = -gdp / sum(abs2, dp)
#     # dp .*= scal
#     # @show scal
#
#     @. nbhd_tmp = dp + g
#
#     # if use_heuristic_neighborhood(cone)
#     #     error("shouldn't be using heuristic nbhd")
#     #     nbhd = norm(nbhd_tmp, Inf) / norm(g, Inf)
#     #     # nbhd = maximum(abs(dj / gj) for (dj, gj) in zip(nbhd_tmp, g)) # TODO try this neighborhood
#     # else
#         has_hess_fact_cache = hasfield(typeof(cone), :hess_fact_cache)
#         if has_hess_fact_cache && !update_hess_fact(cone)
#             return false
#         end
#         nbhd_tmp2 = cone.nbhd_tmp2
#         if has_hess_fact_cache && cone.hess_fact_cache isa DenseSymCache{T}
#             inv_hess_prod!(nbhd_tmp2, nbhd_tmp, cone)
#             nbhd_sqr = dot(nbhd_tmp2, nbhd_tmp)
#             if nbhd_sqr < -eps(T) # TODO possibly loosen
#                 # @warn("numerical failure: cone neighborhood is $nbhd_sqr")
#                 return false
#             end
#             nbhd = sqrt(abs(nbhd_sqr))
#         else
#             inv_hess_sqrt_prod!(nbhd_tmp2, nbhd_tmp, cone)
#             nbhd = norm(nbhd_tmp2)
#         end
#     # end
#
#     # @show nbhd, typeof(cone)
#     # return (nbhd < mu * cone.max_neighborhood)
#     # return (nbhd < 0.5 * mu)
#     # @show nbhd
#     return (nbhd < rtmu * max_nbhd)
#     # return (nbhd < max_nbhd)
#     # return (nbhd < T(0.5))
# end


# utilities for arrays

svec_length(side::Int) = div(side * (side + 1), 2)

svec_idx(row::Int, col::Int) = (div((row - 1) * row, 2) + col)

block_idxs(incr::Int, block::Int) = (incr * (block - 1) .+ (1:incr))

# TODO fix later, rt2::T doesn't work with tests using ForwardDiff
function smat_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{T}, rt2::Number) where {T}
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

function svec_to_smat!(mat::AbstractMatrix{T}, vec::AbstractVector{T}, rt2::Number) where {T}
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

function smat_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{Complex{T}}, rt2::Number) where {T}
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

function svec_to_smat!(mat::AbstractMatrix{Complex{T}}, vec::AbstractVector{T}, rt2::Number) where {T}
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
function rvec_to_cvec!(cvec::AbstractVecOrMat{Complex{T}}, rvec::AbstractVecOrMat{T}) where {T}
    k = 1
    @inbounds for i in eachindex(cvec)
        cvec[i] = Complex(rvec[k], rvec[k + 1])
        k += 2
    end
    return cvec
end

function cvec_to_rvec!(rvec::AbstractVecOrMat{T}, cvec::AbstractVecOrMat{Complex{T}}) where {T}
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
function symm_kron(H::AbstractMatrix{T}, mat::AbstractMatrix{T}, rt2::T) where {T <: Real}
    side = size(mat, 1)
    k = 1
    for i in 1:side, j in 1:i
        k2 = 1
        @inbounds for i2 in 1:side, j2 in 1:i2
            if (i == j) && (i2 == j2)
                H[k2, k] = abs2(mat[i2, i])
            elseif (i != j) && (i2 != j2)
                H[k2, k] = mat[i2, i] * mat[j, j2] + mat[j2, i] * mat[j, i2]
            else
                H[k2, k] = rt2 * mat[i2, i] * mat[j, j2]
            end
            if k2 == k
                break
            end
            k2 += 1
        end
        k += 1
    end
    return H
end

function symm_kron(H::AbstractMatrix{T}, mat::AbstractMatrix{Complex{T}}, rt2::T) where {T <: Real}
    side = size(mat, 1)
    k = 1
    for i in 1:side, j in 1:i
        k2 = 1
        if i == j
            @inbounds for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    H[k2, k] = abs2(mat[i2, i])
                    k2 += 1
                else
                    c = rt2 * mat[i, i2] * mat[j2, j]
                    H[k2, k] = real(c)
                    k2 += 1
                    H[k2, k] = -imag(c)
                    k2 += 1
                end
                if k2 > k
                    break
                end
            end
            k += 1
        else
            @inbounds for i2 in 1:side, j2 in 1:i2
                if i2 == j2
                    c = rt2 * mat[i2, i] * mat[j, j2]
                    H[k2, k] = real(c)
                    H[k2, k + 1] = -imag(c)
                    k2 += 1
                else
                    b1 = mat[i2, i] * mat[j, j2]
                    b2 = mat[j2, i] * mat[j, i2]
                    c1 = b1 + b2
                    H[k2, k] = real(c1)
                    H[k2, k + 1] = -imag(c1)
                    k2 += 1
                    c2 = b1 - b2
                    H[k2, k] = imag(c2)
                    H[k2, k + 1] = real(c2)
                    k2 += 1
                end
                if k2 > k
                    break
                end
            end
            k += 2
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

function sparse_upper_arrow(T::Type{<:Real}, w_dim::Int)
    dim = w_dim + 1
    nnz_tri = 2 * dim - 1
    I = Vector{Int}(undef, nnz_tri)
    J = Vector{Int}(undef, nnz_tri)
    idxs1 = 1:dim
    I[idxs1] .= 1
    J[idxs1] .= idxs1
    idxs2 = (dim + 1):(2 * dim - 1)
    I[idxs2] .= 2:dim
    J[idxs2] .= 2:dim
    V = ones(T, nnz_tri)
    return sparse(I, J, V, dim, dim)
end

function factor_upper_arrow(uu, uw, ww, nzval)
    minval = sqrt(eps(uu)) # TODO tune
    nzidx = 2
    @inbounds for i in eachindex(ww)
        ww1i = ww[i]
        ww1i < eps(uu) && return false
        wwi = sqrt(ww1i)
        uwi = uw[i] / wwi
        uu -= abs2(uwi)
        uu < minval && return false
        nzval[nzidx] = uwi
        nzval[nzidx + 1] = wwi
        nzidx += 2
    end
    nzval[1] = sqrt(uu)
    return true
end

function sparse_upper_arrow_block2(T::Type{<:Real}, w_dim::Int)
    dim = 2 * w_dim + 1
    nnz_tri = 2 * dim - 1 + w_dim
    I = Vector{Int}(undef, nnz_tri)
    J = Vector{Int}(undef, nnz_tri)
    idxs1 = 1:dim
    I[idxs1] .= 1
    J[idxs1] .= idxs1
    idxs2 = (dim + 1):(2 * dim - 1)
    I[idxs2] .= 2:dim
    J[idxs2] .= 2:dim
    idxs3 = (2 * dim):nnz_tri
    I[idxs3] .= 2:2:dim
    J[idxs3] .= 3:2:dim
    V = ones(T, nnz_tri)
    return sparse(I, J, V, dim, dim)
end

function factor_upper_arrow_block2(uu, uv, uw, vv, vw, ww, nzval)
    minval = sqrt(eps(uu)) # TODO tune
    nzidx = 1
    @inbounds for i in eachindex(ww)
        ww1i = ww[i]
        ww1i < eps(uu) && return false
        wwi = sqrt(ww1i)
        vwi = vw[i] / wwi
        uwi = uw[i] / wwi
        vv2i = vv[i] - abs2(vwi)
        vv2i < eps(uu) && return false
        vvi = sqrt(vv2i)
        uvi = (uv[i] - vwi * uwi) / vvi
        uu -= abs2(uwi) + abs2(uvi)
        uu < minval && return false
        @. nzval[nzidx .+ (1:5)] = (uvi, vvi, uwi, vwi, wwi)
        nzidx += 5
    end
    nzval[1] = sqrt(uu)
    return true
end

end
