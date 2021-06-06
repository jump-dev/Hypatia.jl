"""
Proper cone definitions, oracles, and utilities.
"""
module Cones

using DocStringExtensions
using LinearAlgebra
import LinearAlgebra.copytri!
import LinearAlgebra.HermOrSym
import LinearAlgebra.BlasReal
import PolynomialRoots
using SparseArrays
import Hypatia.RealOrComplex
import Hypatia.outer_prod!
import Hypatia.update_eigen!
import Hypatia.spectral_outer!
import Hypatia.posdef_fact_copy!
import Hypatia.inv_fact!

include("arrayutilities.jl")

"""
$(TYPEDEF)

A proper cone.
"""
abstract type Cone{T <: Real} end

"""
$(SIGNATURES)

The real vector dimension of the cone.
"""
dimension(cone::Cone)::Int = cone.dim

"""
$(SIGNATURES)

The barrier parameter ``\\nu`` of the cone.
"""
get_nu(cone::Cone)::Real = cone.nu

"""
$(SIGNATURES)

Set the array equal to the initial interior point for the cone.
"""
function set_initial_point!(arr::AbstractVector, cone::Cone) end

"""
$(SIGNATURES)

Returns true if and only if the currently-loaded primal point is strictly
feasible for the cone.
"""
is_feas(cone::Cone)::Bool = (cone.feas_updated ? cone.is_feas : update_feas(cone))

"""
$(SIGNATURES)

Returns false only if the currently-loaded dual point is outside the interior of
the cone's dual cone.
"""
is_dual_feas(cone::Cone)::Bool = true

"""
$(SIGNATURES)

The gradient of the cone's barrier function at the currently-loaded primal point.
"""
grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))

"""
$(SIGNATURES)

The Hessian (symmetric positive definite) of the cone's barrier function at the
currently-loaded primal point.
"""
function hess(cone::Cone)
    cone.hess_updated && return cone.hess
    return update_hess(cone)
end

"""
$(SIGNATURES)

The inverse Hessian (symmetric positive definite) of the cone's barrier function
at the currently-loaded primal point.
"""
function inv_hess(cone::Cone)
    cone.inv_hess_updated && return cone.inv_hess
    return update_inv_hess(cone)
end

"""
$(SIGNATURES)

Compute the product of the Hessian of the cone's barrier function at the
currently-loaded primal point with a vector or array, in-place.
"""
function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    cone.hess_updated || update_hess(cone)
    mul!(prod, cone.hess, arr)
    return prod
end

"""
$(SIGNATURES)

Compute the product of the inverse Hessian of the cone's barrier function at the
currently-loaded primal point with a vector or array, in-place.
"""
function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    update_hess_fact(cone)
    # TODO try equilibration, iterative refinement etc like posvx/sysvx
    ldiv!(prod, cone.hess_fact, arr)
    return prod
end

"""
$(SIGNATURES)

Returns true if and only if the oracle for the third-order directional
derivative oracle [`dder3`](@ref) can be computed.
"""
use_dder3(::Cone)::Bool = true

"""
$(SIGNATURES)

Compute the third-order directional derivative, in the direction `dir`, the
cone's barrier function at the currently-loaded primal point.
"""
function dder3(cone::Cone, dir::AbstractVector) end

# other oracles and helpers

use_dual_barrier(cone::Cone)::Bool = cone.use_dual_barrier

function setup_data!(cone::Cone{T}) where {T <: Real}
    reset_data(cone)
    dim = dimension(cone)
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    if hasproperty(cone, :dder3)
        cone.dder3 = zeros(T, dim)
    end
    cone.vec1 = zeros(T, dim)
    cone.vec2 = zeros(T, dim)
    setup_extra_data!(cone)
    return cone
end

setup_extra_data!(cone::Cone) = nothing

load_point(
    cone::Cone{T},
    point::AbstractVector{T},
    scal::T,
    ) where {T <: Real} = (@. cone.point = scal * point)

load_point(
    cone::Cone,
    point::AbstractVector,
    ) = copyto!(cone.point, point)

load_dual_point(
    cone::Cone,
    point::AbstractVector,
    ) = copyto!(cone.dual_point, point)

function alloc_hess!(cone::Cone{T}) where {T <: Real}
    dim = dimension(cone)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

function alloc_inv_hess!(cone::Cone{T}) where {T <: Real}
    dim = dimension(cone)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

reset_data(cone::Cone) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

# decide whether to use sqrt oracles
function use_sqrt_hess_oracles(arr_dim::Int, cone::Cone)
    if !cone.hess_fact_updated
        (arr_dim < dimension(cone)) && return false # array is small
        update_hess_fact(cone) || return false
    end
    return (cone.hess_fact isa Cholesky)
end

# only use if use_sqrt_hess_oracles is true
function sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    )
    @assert cone.hess_fact_updated
    mul!(prod, cone.hess_fact.U, arr)
    return prod
end

# only use if use_sqrt_hess_oracles is true
function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    )
    @assert cone.hess_fact_updated
    # TODO try equilibration, iterative refinement etc like posvx/sysvx
    ldiv!(prod, cone.hess_fact.U', arr)
    return prod
end

update_hess_aux(cone::Cone) = nothing

function update_use_hess_prod_slow(cone::Cone{T}) where {T <: Real}
    cone.hess_updated || update_hess(cone)
    @assert cone.hess_updated
    rel_viol = abs(1 - dot(cone.point, cone.hess, cone.point) / get_nu(cone))
    # TODO tune
    cone.use_hess_prod_slow = (rel_viol > dimension(cone) * sqrt(eps(T)))
    # cone.use_hess_prod_slow && println("switching to slow hess prod")
    cone.use_hess_prod_slow_updated = true
    return
end

hess_prod_slow!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    ) = hess_prod!(prod, arr, cone)

function update_hess_fact(cone::Cone{T}) where {T <: Real}
    cone.hess_fact_updated && return true
    cone.hess_updated || update_hess(cone)
    if !isdefined(cone, :hess_fact_mat)
        cone.hess_fact_mat = zero(cone.hess)
    end

    # do not modify the hessian during recovery
    cone.hess_fact = posdef_fact_copy!(cone.hess_fact_mat, cone.hess, false)

    cone.hess_fact_updated = true
    return issuccess(cone.hess_fact)
end

function update_inv_hess(cone::Cone)
    isdefined(cone, :inv_hess) || alloc_inv_hess!(cone)
    update_hess_fact(cone)
    inv_fact!(cone.inv_hess.data, cone.hess_fact)
    cone.inv_hess_updated = true
    return cone.inv_hess
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

# check numerics of some oracles used in proximity check TODO tune
function check_numerics(
    cone::Cone{T},
    gtol::T = sqrt(sqrt(eps(T))),
    Htol::T = 10sqrt(gtol),
    ) where {T <: Real}
    g = grad(cone)
    dim = length(g)
    nu = get_nu(cone)

    # grad check
    (abs(1 + dot(g, cone.point) / nu) > gtol * dim) && return false

    # inv hess check
    Hig = inv_hess_prod!(cone.vec1, g, cone)
    (abs(1 - dot(Hig, g) / nu) > Htol * dim) && return false

    return true
end

# compute central path proximity for a cone; if using max proximity, proximity
# is computed differently if cone is not primitive, eg nonnegative cone
function get_proximity(
    cone::Cone{T},
    rtmu::T,
    ::Bool, # use sum proximity
    negtol::T = sqrt(eps(T)),
    ) where {T <: Real}
    g = grad(cone)
    vec1 = cone.vec1
    vec2 = cone.vec2

    @. vec1 = cone.dual_point + rtmu * g
    inv_hess_prod!(vec2, vec1, cone)
    prox_sqr = dot(vec2, vec1)
    (prox_sqr < -negtol * length(g)) && return T(NaN) # should be positive
    return sqrt(abs(prox_sqr)) / rtmu
end

include("nonnegative.jl")
include("possemideftri.jl")
include("doublynonnegativetri.jl")
include("possemideftrisparse/possemideftrisparse.jl")
include("linmatrixineq.jl")
include("epinorminf.jl")
include("epinormeucl.jl")
include("epipersquare.jl")
include("epinormspectral.jl")
include("matrixepipersquare.jl")
include("generalizedpower.jl")
include("hypopowermean.jl")
include("hypogeomean.jl")
include("hyporootdettri.jl")
include("hypoperlog.jl")
include("hypoperlogdettri.jl")
include("epipersepspectral/epipersepspectral.jl")
include("epirelentropy.jl")
include("epitrrelentropytri.jl")
include("wsosinterpnonnegative.jl")
include("wsosinterppossemideftri.jl")
include("wsosinterpepinormone.jl")
include("wsosinterpepinormeucl.jl")

end
