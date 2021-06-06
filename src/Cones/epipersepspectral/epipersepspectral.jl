"""
$(TYPEDEF)

A cone of squares on a Jordan algebra.
"""
abstract type ConeOfSquares{T <: Real} end

# a cache for a cone of squares implementation
abstract type CSqrCache{T <: Real} end

"""
$(TYPEDEF)

A univariate convex function defined on positive reals.
"""
abstract type SepSpectralFun end

"""
$(TYPEDEF)

Epigraph of perspective function of a convex separable spectral function `h`
over a cone of squares `Q` on a Jordan algebra with rank `d`.

    $(FUNCTIONNAME){Q, T}(h::Hypatia.Cones.SepSpectralFun, d::Int, use_dual::Bool = false)
"""
mutable struct EpiPerSepSpectral{Q <: ConeOfSquares, T <: Real} <: Cone{T}
    h::SepSpectralFun
    use_dual_barrier::Bool
    d::Int
    dim::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    dder3::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_aux_updated::Bool
    inv_hess_aux_updated::Bool
    dder3_aux_updated::Bool
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_mat::Symmetric{T, Matrix{T}}
    hess_fact::Factorization{T}

    w_view::SubArray{T, 1}
    cache::CSqrCache{T}

    function EpiPerSepSpectral{Q, T}(
        h::SepSpectralFun,
        d::Int; # dimension/rank parametrizing the cone of squares
        use_dual::Bool = false,
        ) where {T <: Real, Q <: ConeOfSquares{T}}
        @assert d >= 1
        cone = new{Q, T}()
        cone.h = h
        cone.use_dual_barrier = use_dual
        cone.d = d
        cone.dim = 2 + vector_dim(Q, d)
        return cone
    end
end

reset_data(cone::EpiPerSepSpectral) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated =
    cone.inv_hess_aux_updated = cone.dder3_aux_updated =
    cone.hess_fact_updated = false)

function setup_extra_data!(cone::EpiPerSepSpectral)
    @views cone.w_view = cone.point[3:end]
    setup_csqr_cache(cone)
    return cone
end

get_nu(cone::EpiPerSepSpectral) = 2 + cone.d

include("vectorcsqr.jl")
include("matrixcsqr.jl")

include("sepspectralfun.jl")
