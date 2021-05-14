#=
(closure of) epigraph of perspective of trace of a generic separable spectral
function h over a cone of squares Q on a Jordan algebra:
    (u, v, w) in ℝ × ℝ₊₊ × Q :
    u ≥ v h(w / v) = ∑ᵢ v h(λᵢ(w / v))
=#

# type of cone of squares on a Jordan algebra
abstract type ConeOfSquares{T <: Real} end

# cache for cone of squares oracles implementation
abstract type CSqrCache{T <: Real} end

# suitable univariate matrix monotone function
abstract type SepSpectralFun end

mutable struct EpiPerSepSpectral{Q <: ConeOfSquares, T <: Real} <: Cone{T}
    h::SepSpectralFun
    use_dual_barrier::Bool
    d::Int
    dim::Int
    nu::Int

    point::Vector{T}
    dual_point::Vector{T}
    grad::Vector{T}
    correction::Vector{T}
    vec1::Vector{T}
    vec2::Vector{T}
    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    correction_updated::Bool
    hess_aux_updated::Bool
    inv_hess_aux_updated::Bool
    correction_aux_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    w_view
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
        cone.nu = 2 + d
        return cone
    end
end

reset_data(cone::EpiPerSepSpectral) = (cone.feas_updated = cone.grad_updated =
    cone.hess_updated = cone.inv_hess_updated = cone.hess_aux_updated =
    cone.inv_hess_aux_updated = cone.correction_updated =
    cone.correction_aux_updated = false)

use_sqrt_hess_oracles(cone::EpiPerSepSpectral) = false

function setup_extra_data(
    cone::EpiPerSepSpectral{<:ConeOfSquares{T}},
    ) where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    @views cone.w_view = cone.point[3:end]
    setup_csqr_cache(cone)
    return cone
end

include("vectorcsqr.jl")
include("matrixcsqr.jl")

include("sepspectralfun.jl")
