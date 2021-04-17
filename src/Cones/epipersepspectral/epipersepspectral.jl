#=
TODO

(closure of) epigraph of perspective of tr of a generic separable spectral function on a cone of squares on a Jordan algebra
=#

# type of cone of squares on a Jordan algebra
abstract type ConeOfSquares{T <: Real} end

# cache for cone of squares oracles implementation
abstract type CSqrCache{T <: Real} end

# suitable univariate matrix monotone function
abstract type SepSpectralFun end

# TODO maybe don't need F as a type parameter (may slow down compile), could just be a field - decide later
mutable struct EpiPerSepSpectral{Q <: ConeOfSquares, F <: SepSpectralFun, T <: Real} <: Cone{T}
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
    hess_fact_updated::Bool
    is_feas::Bool
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    w_view
    cache::CSqrCache{T}

    function EpiPerSepSpectral{Q, F, T}(
        d::Int; # dimension parametrizing the cone of squares (not vectorized dimension)
        use_dual::Bool = false,
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real, Q <: ConeOfSquares{T}, F <: SepSpectralFun}
        @assert d >= 1
        cone = new{Q, F, T}()
        cone.use_dual_barrier = use_dual
        cone.d = d
        cone.dim = 2 + vector_dim(Q, d)
        cone.nu = 2 + d
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

function setup_extra_data(cone::EpiPerSepSpectral{<:ConeOfSquares{T}}) where {T <: Real}
    dim = cone.dim
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    @views cone.w_view = cone.point[3:end]
    setup_csqr_cache(cone)
    return cone
end

include("vectorcsqr.jl")
include("matrixcsqr.jl")
# include("epinormcsqr.jl")
# const ConeOfSquaresList = [
#     VectorCSqr,
#     MatrixCSqr,
#     # EpiNormCSqr,
#     ]

include("sepspectralfun.jl")
# const SepSpectralFunList = [
#     NegLogMMF,
#     EntropyMMF,
#     Power12MMF,
#     ]
