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
# include("epiperpower.jl")
# include("hypoperlog.jl")
# include("hypogeomean.jl")
# include("epinormspectral.jl")
# include("hypopersumlog.jl")
# include("epipersumexp.jl")
include("semidefinite.jl")
# include("hypoperlogdet.jl")
include("wsospolyinterp.jl")
# include("wsospolyinterpmat.jl")
# include("wsospolyinterpsoc.jl")

use_dual(cone::Cone) = cone.use_dual
load_point(cone::Cone, point::AbstractVector) = (reset_data(cone); cone.point = point)
dimension(cone::Cone) = cone.dim

update_hess_prod(cone::Cone) = nothing
update_inv_hess_prod(cone::Cone) = nothing

is_feas(cone::Cone) = (cone.feas_updated ? cone.is_feas : update_feas(cone))
grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))
hess(cone::Cone) = (cone.hess_updated ? cone.hess : update_hess(cone))
inv_hess(cone::Cone) = (cone.inv_hess_updated ? cone.inv_hess : update_inv_hess(cone))


# mutable struct ProductCone{T <: HypReal} end
#     cones::Vector{Cone{T}}
#     num_cones::Int
#     dim::Int
#     bar_par::T
#
#     feas_list::Vector{Bool}
#     grad_list::Vector{Vector{T}}
#     hess_list::Vector
#     inv_hess_list::Vector
#
#     function ProductCone{T}(cones::Vector{Cone{T}}) where {T <: HypReal}
#         prodcone = new{T}()
#         prodcone.cones = cones
#         prodcone.num_cones = length(cones)
#         prodcone.dim = sum(...)
#         prodcone.bar_par = sum(...)
#         # TODO allocate these extra algorithmic fields only at start of algorithm?
#         prodcone.feas_list = falses(prodcone.num_cones)
#         prodcone.grad_list = Vector{Vector{T}}(undef, prodcone.num_cones)
#         prodcone.hess_list = Vector{Any}(undef, prodcone.num_cones)
#         prodcone.inv_hess_list = Vector{Any}(undef, prodcone.num_cones)
#         return prodcone
#     end
# end


# function grad(cone::Cone)






# function factorize_hess(cone::Cone)
#     copyto!(cone.H2, cone.H)
#     cone.F = hyp_chol!(Symmetric(cone.H2, :U))
#     return isposdef(cone.F)
# end

# grad(cone::Cone) = cone.g
# hess(cone::Cone) = Symmetric(cone.H, :U)
# inv_hess(cone::Cone) = Symmetric(inv(cone.F), :U)
# hess_fact(cone::Cone) = cone.F

# function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
#     @assert cone.hess_updated
#     return mul!(prod, cone.hess, arr)
# end
#
# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
#     @assert cone.inv_hess_updated
#     return ldiv!(prod, cone.F, arr)
# end

# utilities for converting between smat and svec forms (lower triangle) for symmetric matrices
# TODO only need to do lower triangle if use symmetric matrix types

function smat_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{T}, rt2::T) where {T <: HypReal}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            vec[k] = mat[i, j]
        else
            vec[k] = rt2 * mat[i, j]
        end
        k += 1
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix{T}, vec::AbstractVector{T}, rt2i::T) where {T <: HypReal}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            mat[i, j] = vec[k]
        else
            mat[i, j] = mat[j, i] = rt2i * vec[k]
        end
        k += 1
    end
    return mat
end

function smat_to_svec!(vec::AbstractVector{T}, mat::AbstractMatrix{Complex{T}}, rt2::T) where {T <: HypReal}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            vec[k] = real(mat[i, j])
            k += 1
        else
            ck = rt2 * mat[i, j]
            vec[k] = real(ck)
            k += 1
            vec[k] = imag(ck)
            k += 1
        end
    end
    return vec
end

function svec_to_smat!(mat::AbstractMatrix{Complex{T}}, vec::AbstractVector{T}, rt2i::T) where {T <: HypReal}
    k = 1
    m = size(mat, 1)
    for i in 1:m, j in 1:i
        if i == j
            mat[i, j] = vec[k]
            k += 1
        else
            mat[i, j] = mat[j, i] = rt2i * Complex(vec[k], vec[k + 1])
            k += 2
        end
    end
    return mat
end

end
