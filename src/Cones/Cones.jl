#=
Copyright 2018, Chris Coey and contributors

functions and caches for cones
=#

module Cones

using LinearAlgebra
import LinearAlgebra.BlasFloat
using ForwardDiff
using DiffResults
import Hypatia.HypReal
import Hypatia.HypRealOrComplex
import Hypatia.hyp_AtA!
import Hypatia.hyp_chol!
import Hypatia.hyp_ldiv_chol_L!

abstract type Cone{T <: HypReal} end

include("orthant.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epipersquare.jl")
include("epiperpower.jl")
include("hypoperlog.jl")
include("epiperexp.jl")
include("hypogeomean.jl")
include("epinormspectral.jl")
include("semidefinite.jl")
include("hypoperlogdet.jl")
include("wsospolyinterp.jl")
# include("wsospolyinterpmat.jl")
# include("wsospolyinterpsoc.jl")

use_dual(cone::Cone) = cone.use_dual
load_point(cone::Cone, point::AbstractVector) = (cone.point = point)
dimension(cone::Cone) = cone.dim

is_feas(cone::Cone) = (cone.feas_updated ? cone.is_feas : update_feas(cone))
grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))
hess(cone::Cone) = (cone.hess_updated ? cone.hess : update_hess(cone))
inv_hess(cone::Cone) = (cone.inv_hess_updated ? cone.inv_hess : update_inv_hess(cone))

# fallbacks

reset_data(cone::Cone) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.inv_hess_prod_updated = false)

# TODO check if this is most efficient way to use DiffResults
function update_grad(cone::Cone)
    @assert cone.is_feas
    cone.diffres = ForwardDiff.hessian!(cone.diffres, cone.barfun, cone.point)
    cone.grad .= DiffResults.gradient(cone.diffres)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::Cone)
    @assert cone.grad_updated
    cone.hess.data .= DiffResults.hessian(cone.diffres)
    cone.hess_updated = true
    return cone.hess
end

update_hess_prod(cone::Cone) = nothing

function update_inv_hess_prod(cone::Cone)
    if !cone.hess_updated
        update_hess(cone)
    end
    copyto!(cone.tmp_hess, cone.hess)
    cone.hess_fact = hyp_chol!(cone.tmp_hess)
    if !isposdef(cone.hess_fact) # TODO maybe better to not step to this point if the hessian factorization fails
        copyto!(cone.tmp_hess, cone.hess)
        cone.hess_fact = lu!(cone.tmp_hess)
    end
    cone.inv_hess_prod_updated = true
    return
end

function update_inv_hess(cone::Cone)
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    cone.inv_hess = Symmetric(inv(cone.hess_fact), :U)
    cone.inv_hess_updated = true
    return cone.inv_hess
end

# TODO maybe write using linear operator form rather than needing explicit hess
function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    if !cone.hess_updated
        update_hess(cone)
    end
    return mul!(prod, cone.hess, arr)
end

function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    if !cone.inv_hess_prod_updated
        update_inv_hess_prod(cone)
    end
    return ldiv!(prod, cone.hess_fact, arr)
end

# utilities for converting between smat and svec forms (lower triangle) for symmetric matrices

function smat_L_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{T}) where {T}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            vec[k] = mat[i, j]
        else
            vec[k] = 2 * mat[i, j]
        end
        k += 1
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix{T}, vec::AbstractVector{T}) where {T}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            mat[i, j] = vec[k]
        else
            mat[i, j] = mat[j, i] = vec[k] / 2
        end
        k += 1
    end
    return mat
end

function smat_L_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{Complex{T}}) where {T}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            vec[k] = real(mat[i, j])
            k += 1
        else
            ck = 2 * mat[i, j]
            vec[k] = real(ck)
            k += 1
            vec[k] = imag(ck)
            k += 1
        end
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix{Complex{T}}, vec::AbstractVector{T}) where {T}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            mat[i, j] = vec[k]
            k += 1
        else
            mat[i, j] = Complex(vec[k], vec[k + 1]) / 2
            mat[j, i] = Complex(vec[k], -vec[k + 1]) / 2
            k += 2
        end
    end
    return mat
end

function mat_L_to_vec!(vec::AbstractVector{T}, mat::AbstractMatrix{T}) where {T}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        vec[k] = mat[i, j]
        k += 1
    end
    return vec
end

function vec_to_mat!(mat::AbstractMatrix{T}, vec::AbstractVector{T}) where {T}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        mat[i, j] = mat[j, i] = vec[k]
        k += 1
    end
    return mat
end

function mat_L_to_vec!(vec::AbstractVector{T}, mat::AbstractMatrix{Complex{T}}) where {T}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            vec[k] = real(mat[i, j])
            k += 1
        else
            ck = mat[i, j]
            vec[k] = real(ck)
            k += 1
            vec[k] = imag(ck)
            k += 1
        end
    end
    return vec
end

function vec_to_mat!(mat::AbstractMatrix{Complex{T}}, vec::AbstractVector{T}) where {T}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            mat[i, j] = vec[k]
            k += 1
        else
            mat[i, j] = Complex(vec[k], vec[k + 1])
            mat[j, i] = Complex(vec[k], -vec[k + 1])
            k += 2
        end
    end
    return mat
end

end
