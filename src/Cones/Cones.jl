#=
Copyright 2018, Chris Coey and contributors

functions and caches for cones
=#

module Cones

using LinearAlgebra
import LinearAlgebra.copytri!
import Hypatia.RealOrComplex
import Hypatia.DenseSymCache
import Hypatia.DensePosDefCache
import Hypatia.load_matrix
import Hypatia.update_fact
import Hypatia.inv_prod
import Hypatia.sqrt_prod
import Hypatia.inv_sqrt_prod
import Hypatia.invert

# hessian_cache(T::Type{<:LinearAlgebra.BlasReal}) = DenseSymCache{T}() # use Bunch Kaufman for BlasReals from start
hessian_cache(T::Type{<:Real}) = DensePosDefCache{T}()

abstract type Cone{T <: Real} end

include("nonnegative.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epipersquare.jl")
include("power.jl")
include("hypoperlog.jl")
include("epiperexp.jl")
include("hypogeomean.jl")
include("epinormspectral.jl")
include("possemideftri.jl")
include("hypoperlogdettri.jl")
include("hyporootdettri.jl")
include("wsosinterpnonnegative.jl")
include("wsosinterppossemideftri.jl")
include("wsosinterpepinormeucl.jl")

use_dual(cone::Cone) = cone.use_dual
load_point(cone::Cone, point::AbstractVector) = copyto!(cone.point, point)
dimension(cone::Cone) = cone.dim

is_feas(cone::Cone) = (cone.feas_updated ? cone.is_feas : update_feas(cone))
grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))
hess(cone::Cone) = (cone.hess_updated ? cone.hess : update_hess(cone))
inv_hess(cone::Cone) = (cone.inv_hess_updated ? cone.inv_hess : update_inv_hess(cone))

# fallbacks

reset_data(cone::Cone) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

update_hess_prod(cone::Cone) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    cone.hess_updated || update_hess(cone)
    return mul!(prod, cone.hess, arr)
end

function update_hess_fact(cone::Cone{T}) where {T <: Real}
    @assert !cone.hess_fact_updated
    cone.hess_updated || update_hess(cone)

    if !update_fact(cone.hess_fact_cache, cone.hess)
        if T <: LinearAlgebra.BlasReal && cone.hess_fact_cache isa DensePosDefCache{T}
            @warn("Switching Hessian cache from Cholesky to Bunch Kaufman")
            cone.hess_fact_cache = DenseSymCache{T}()
            load_matrix(cone.hess_fact_cache, cone.hess)
        else
            cone.hess += sqrt(eps(T)) * I # attempt recovery # TODO make more efficient
        end
        if !update_fact(cone.hess_fact_cache, cone.hess)
            @warn("Hessian factorization failed after recovery")
            return false
        end
    end

    cone.hess_fact_updated = true
    return true
end

function update_inv_hess(cone::Cone)
    @assert !cone.inv_hess_updated
    cone.hess_fact_updated || update_hess_fact(cone)
    invert(cone.hess_fact_cache, cone.inv_hess)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    cone.hess_fact_updated || update_hess_fact(cone)
    copyto!(prod, arr)
    return inv_prod(cone.hess_fact_cache, prod) # TODO change signature to match sqrt_prod
end

function hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    cone.hess_fact_updated || update_hess_fact(cone)
    return sqrt_prod(cone.hess_fact_cache, prod, arr)
end

function inv_hess_sqrt_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    cone.hess_fact_updated || update_hess_fact(cone)
    return inv_sqrt_prod(cone.hess_fact_cache, prod, arr)
end

# fallbacks for sparse linear system solvers
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
function rvec_to_cvec!(cvec::AbstractVector{Complex{T}}, rvec::AbstractVector{T}) where {T}
    k = 1
    @inbounds for i in eachindex(cvec)
        cvec[i] = Complex(rvec[k], rvec[k + 1])
        k += 2
    end
    return cvec
end

function cvec_to_rvec!(rvec::AbstractVector{T}, cvec::AbstractVector{Complex{T}}) where {T}
    k = 1
    @inbounds for i in eachindex(cvec)
        ci = cvec[i]
        rvec[k] = real(ci)
        rvec[k + 1] = imag(ci)
        k += 2
    end
    return rvec
end

vec_copy_to!(v1::AbstractVector{T}, v2::AbstractVector{T}) where {T <: Real} = copyto!(v1, v2)
vec_copy_to!(v1::AbstractVector{T}, v2::AbstractVector{Complex{T}}) where {T <: Real} = cvec_to_rvec!(v1, v2)
vec_copy_to!(v1::AbstractVector{Complex{T}}, v2::AbstractVector{T}) where {T <: Real} = rvec_to_cvec!(v1, v2)

end
