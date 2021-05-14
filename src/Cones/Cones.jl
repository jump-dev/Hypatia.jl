#=
functions and caches for cones
=#

module Cones

using LinearAlgebra
import LinearAlgebra.copytri!
import LinearAlgebra.HermOrSym
import LinearAlgebra.BlasReal
import PolynomialRoots
using SparseArrays
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

include("arrayutilities.jl")

abstract type Cone{T <: Real} end

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

use_dual_barrier(cone::Cone) = cone.use_dual_barrier
dimension(cone::Cone) = cone.dim

function setup_data(cone::Cone{T}) where {T <: Real}
    reset_data(cone)
    dim = dimension(cone)
    cone.point = zeros(T, dim)
    cone.dual_point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    if hasproperty(cone, :correction)
        cone.correction = zeros(T, dim)
    end
    cone.vec1 = zeros(T, dim)
    cone.vec2 = zeros(T, dim)
    setup_extra_data(cone)
    return cone
end

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

is_feas(cone::Cone) = (cone.feas_updated ? cone.is_feas : update_feas(cone))
is_dual_feas(cone::Cone) = true # use neighborhood check for dual feasibility

grad(cone::Cone) = (cone.grad_updated ? cone.grad : update_grad(cone))

hess(cone::Cone) = (cone.hess_updated ? cone.hess : update_hess(cone))

inv_hess(cone::Cone) = (cone.inv_hess_updated ? cone.inv_hess :
    update_inv_hess(cone))

# fallbacks

# hessian_cache(T::Type{<:BlasReal}) = DenseSymCache{T}() # BunchKaufman for BlasReal
hessian_cache(T::Type{<:Real}) = DensePosDefCache{T}()

reset_data(cone::Cone) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_fact_updated = false)

get_nu(cone::Cone) = cone.nu

function use_sqrt_hess_oracles(cone::Cone)
    if !cone.hess_fact_updated
        update_hess_fact(cone) || return false
    end
    return (cone.hess_fact_cache isa DensePosDefCache)
end

use_correction(::Cone) = true

update_hess_aux(cone::Cone) = nothing

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::Cone)
    cone.hess_updated || update_hess(cone)
    mul!(prod, cone.hess, arr)
    return prod
end

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

function inv_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    )
    update_hess_fact(cone)
    copyto!(prod, arr)
    inv_prod(cone.hess_fact_cache, prod)
    return prod
end

function sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    )
    update_hess_fact(cone)
    copyto!(prod, arr)
    sqrt_prod(cone.hess_fact_cache, prod)
    return prod
end

function inv_sqrt_hess_prod!(
    prod::AbstractVecOrMat,
    arr::AbstractVecOrMat,
    cone::Cone,
    )
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

function in_neighborhood(
    cone::Cone{T},
    rtmu::T,
    max_nbhd::T;
    # use_heuristic_neighborhood::Bool = false, # TODO make option to solver
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
    (abs(1 + dot(g, cone.point) / nu) > gtol * dim) && return false
    # inv hess check
    inv_hess_prod!(vec1, g, cone)
    (abs(1 - dot(vec1, g) / nu) > Htol * dim) && return false

    # check neighborhood condition
    @. vec1 = cone.dual_point + rtmu * g
    # if use_heuristic_neighborhood(cone)
    #     nbhd = norm(vec1, Inf) / norm(g, Inf)
    # else
        vec2 = cone.vec2
        inv_hess_prod!(vec2, vec1, cone)
        nbhd_sqr = dot(vec2, vec1)
        if nbhd_sqr < -tol * dim
            return false
        end
        nbhd = sqrt(abs(nbhd_sqr))
    # end

    return (nbhd < rtmu * max_nbhd)
end

end
