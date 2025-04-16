#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
definitions of conic sets not already defined by MathOptInterface
and functions for converting between Hypatia and MOI cone definitions
=#

function cone_from_moi(::Type{<:Real}, cone::MOI.AbstractVectorSet)
    return error("MOI set $cone is not recognized")
end

# MOI predefined cones

function cone_from_moi(::Type{T}, cone::MOI.Nonnegatives) where {T <: Real}
    return Cones.Nonnegative{T}(MOI.dimension(cone))
end

function cone_from_moi(
    ::Type{T},
    cone::MOI.PositiveSemidefiniteConeTriangle,
) where {T <: Real}
    return Cones.PosSemidefTri{T, T}(MOI.dimension(cone))
end

function cone_from_moi(
    ::Type{T},
    cone::MOI.HermitianPositiveSemidefiniteConeTriangle,
) where {T <: Real}
    return Cones.PosSemidefTri{T, Complex{T}}(MOI.dimension(cone))
end

function cone_from_moi(::Type{T}, cone::MOI.NormInfinityCone) where {T <: Real}
    return Cones.EpiNormInf{T, T}(MOI.dimension(cone))
end

function cone_from_moi(::Type{T}, cone::MOI.NormOneCone) where {T <: Real}
    return Cones.EpiNormInf{T, T}(MOI.dimension(cone), use_dual = true)
end

function cone_from_moi(::Type{T}, cone::MOI.SecondOrderCone) where {T <: Real}
    return Cones.EpiNormEucl{T}(MOI.dimension(cone))
end

function cone_from_moi(::Type{T}, cone::MOI.RotatedSecondOrderCone) where {T <: Real}
    return Cones.EpiPerSquare{T}(MOI.dimension(cone))
end

function cone_from_moi(::Type{T}, cone::MOI.NormSpectralCone) where {T <: Real}
    return Cones.EpiNormSpectral{T, T}(extrema((cone.row_dim, cone.column_dim))...)
end

function cone_from_moi(::Type{T}, cone::MOI.NormNuclearCone) where {T <: Real}
    return Cones.EpiNormSpectral{T, T}(
        extrema((cone.row_dim, cone.column_dim))...,
        use_dual = true,
    )
end

function cone_from_moi(::Type{T}, cone::MOI.PowerCone{T}) where {T <: Real}
    return Cones.GeneralizedPower{T}(T[cone.exponent, 1 - cone.exponent], 1)
end

function cone_from_moi(::Type{T}, cone::MOI.DualPowerCone{T}) where {T <: Real}
    return Cones.GeneralizedPower{T}(
        T[cone.exponent, 1 - cone.exponent],
        1,
        use_dual = true,
    )
end

function cone_from_moi(::Type{T}, cone::MOI.GeometricMeanCone) where {T <: Real}
    return (l = MOI.dimension(cone) - 1; Cones.HypoGeoMean{T}(1 + l))
end

function cone_from_moi(::Type{T}, cone::MOI.RootDetConeTriangle) where {T <: Real}
    return Cones.HypoRootdetTri{T, T}(MOI.dimension(cone))
end

function cone_from_moi(::Type{T}, cone::MOI.ExponentialCone) where {T <: Real}
    return Cones.HypoPerLog{T}(3)
end

function cone_from_moi(::Type{T}, cone::MOI.DualExponentialCone) where {T <: Real}
    return Cones.HypoPerLog{T}(3, use_dual = true)
end

function cone_from_moi(::Type{T}, cone::MOI.LogDetConeTriangle) where {T <: Real}
    return Cones.HypoPerLogdetTri{T, T}(MOI.dimension(cone))
end

function cone_from_moi(::Type{T}, cone::MOI.RelativeEntropyCone) where {T <: Real}
    return Cones.EpiRelEntropy{T}(MOI.dimension(cone))
end

# Hypatia predefined cones
# some are equivalent to above MOI predefined cones, but we define again for the sake of consistency

"""
$(TYPEDEF)

See [`Cones.Nonnegative`](@ref).

$(TYPEDFIELDS)
"""
struct NonnegativeCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
end
export NonnegativeCone

MOI.dimension(cone::NonnegativeCone) = cone.dim

function cone_from_moi(::Type{T}, cone::NonnegativeCone{T}) where {T <: Real}
    return Cones.Nonnegative{T}(cone.dim)
end

"""
$(TYPEDEF)

See [`Cones.PosSemidefTri`](@ref).

$(TYPEDFIELDS)
"""
struct PosSemidefTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
end
export PosSemidefTriCone

MOI.dimension(cone::PosSemidefTriCone) = cone.dim

function cone_from_moi(
    ::Type{T},
    cone::PosSemidefTriCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return Cones.PosSemidefTri{T, R}(cone.dim)
end

"""
$(TYPEDEF)

See [`Cones.DoublyNonnegativeTri`](@ref).

$(TYPEDFIELDS)
"""
struct DoublyNonnegativeTriCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
export DoublyNonnegativeTriCone

function DoublyNonnegativeTriCone{T}(dim::Int) where {T <: Real}
    return DoublyNonnegativeTriCone{T}(dim, false)
end

MOI.dimension(cone::DoublyNonnegativeTriCone where {T <: Real}) = cone.dim

function cone_from_moi(::Type{T}, cone::DoublyNonnegativeTriCone{T}) where {T <: Real}
    return Cones.DoublyNonnegativeTri{T}(cone.dim, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.PosSemidefTriSparse`](@ref).

$(TYPEDFIELDS)
"""
struct PosSemidefTriSparseCone{
    I <: Cones.PSDSparseImpl,
    T <: Real,
    R <: RealOrComplex{T},
} <: MOI.AbstractVectorSet
    side::Int
    row_idxs::Vector{Int}
    col_idxs::Vector{Int}
    use_dual::Bool
end
export PosSemidefTriSparseCone

function PosSemidefTriSparseCone{I, T, R}(
    side::Int,
    row_idxs::Vector{Int},
    col_idxs::Vector{Int},
) where {I <: Cones.PSDSparseImpl, T <: Real, R <: RealOrComplex{T}}
    return PosSemidefTriSparseCone{I, T, R}(side, row_idxs, col_idxs, false)
end

function MOI.dimension(
    cone::PosSemidefTriSparseCone{<:Cones.PSDSparseImpl, T, T} where {T <: Real},
)
    return length(cone.row_idxs)
end

function MOI.dimension(
    cone::PosSemidefTriSparseCone{<:Cones.PSDSparseImpl, T, Complex{T}} where {T <: Real},
)
    return (2 * length(cone.row_idxs) - cone.side)
end

function cone_from_moi(
    ::Type{T},
    cone::PosSemidefTriSparseCone{I, T, R},
) where {I <: Cones.PSDSparseImpl, T <: Real, R <: RealOrComplex{T}}
    return Cones.PosSemidefTriSparse{I, T, R}(
        cone.side,
        cone.row_idxs,
        cone.col_idxs,
        use_dual = cone.use_dual,
    )
end

"""
$(TYPEDEF)

See [`Cones.LinMatrixIneq`](@ref).

$(TYPEDFIELDS)
"""
struct LinMatrixIneqCone{T <: Real} <: MOI.AbstractVectorSet
    As::Vector
    use_dual::Bool
end
export LinMatrixIneqCone

LinMatrixIneqCone{T}(As::Vector) where {T <: Real} = LinMatrixIneqCone{T}(As, false)

MOI.dimension(cone::LinMatrixIneqCone) = length(cone.As)

function cone_from_moi(::Type{T}, cone::LinMatrixIneqCone{T}) where {T <: Real}
    return Cones.LinMatrixIneq{T}(cone.As, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.EpiNormInf`](@ref).

$(TYPEDFIELDS)
"""
struct EpiNormInfCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
export EpiNormInfCone

function EpiNormInfCone{T, R}(dim::Int) where {T <: Real, R <: RealOrComplex{T}}
    return EpiNormInfCone{T, R}(dim, false)
end

MOI.dimension(cone::EpiNormInfCone) = cone.dim

function cone_from_moi(
    ::Type{T},
    cone::EpiNormInfCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return Cones.EpiNormInf{T, R}(cone.dim, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.EpiNormEucl`](@ref).

$(TYPEDFIELDS)
"""
struct EpiNormEuclCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
end
export EpiNormEuclCone

MOI.dimension(cone::EpiNormEuclCone) = cone.dim

function cone_from_moi(::Type{T}, cone::EpiNormEuclCone{T}) where {T <: Real}
    return Cones.EpiNormEucl{T}(cone.dim)
end

"""
$(TYPEDEF)

See [`Cones.EpiPerSquare`](@ref).

$(TYPEDFIELDS)
"""
struct EpiPerSquareCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
end
export EpiPerSquareCone

MOI.dimension(cone::EpiPerSquareCone) = cone.dim

function cone_from_moi(::Type{T}, cone::EpiPerSquareCone{T}) where {T <: Real}
    return Cones.EpiPerSquare{T}(cone.dim)
end

"""
$(TYPEDEF)

See [`Cones.EpiNormSpectralTri`](@ref).

$(TYPEDFIELDS)
"""
struct EpiNormSpectralTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
export EpiNormSpectralTriCone

function EpiNormSpectralTriCone{T, R}(dim::Int) where {T <: Real, R <: RealOrComplex{T}}
    return EpiNormSpectralTriCone{T, R}(dim, false)
end

MOI.dimension(cone::EpiNormSpectralTriCone) = cone.dim

function cone_from_moi(
    ::Type{T},
    cone::EpiNormSpectralTriCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return Cones.EpiNormSpectralTri{T, R}(cone.dim, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.EpiNormSpectral`](@ref).

$(TYPEDFIELDS)
"""
struct EpiNormSpectralCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    d1::Int
    d2::Int
    use_dual::Bool
end
export EpiNormSpectralCone

function EpiNormSpectralCone{T, R}(
    d1::Int,
    d2::Int,
) where {T <: Real, R <: RealOrComplex{T}}
    return EpiNormSpectralCone{T, R}(d1, d2, false)
end

function MOI.dimension(
    cone::EpiNormSpectralCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return 1 + Cones.vec_length(R, cone.d1 * cone.d2)
end

function cone_from_moi(
    ::Type{T},
    cone::EpiNormSpectralCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return Cones.EpiNormSpectral{T, R}(cone.d1, cone.d2, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.MatrixEpiPerSquare`](@ref).

$(TYPEDFIELDS)
"""
struct MatrixEpiPerSquareCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    d1::Int
    d2::Int
    use_dual::Bool
end
export MatrixEpiPerSquareCone

function MatrixEpiPerSquareCone{T, R}(
    d1::Int,
    d2::Int,
) where {T <: Real, R <: RealOrComplex{T}}
    return MatrixEpiPerSquareCone{T, R}(d1, d2, false)
end

function MOI.dimension(
    cone::MatrixEpiPerSquareCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return Cones.svec_length(R, cone.d1) + 1 + Cones.vec_length(R, cone.d1 * cone.d2)
end

function cone_from_moi(
    ::Type{T},
    cone::MatrixEpiPerSquareCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return Cones.MatrixEpiPerSquare{T, R}(cone.d1, cone.d2, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.GeneralizedPower`](@ref).

$(TYPEDFIELDS)
"""
struct GeneralizedPowerCone{T <: Real} <: MOI.AbstractVectorSet
    α::Vector{T}
    n::Int
    use_dual::Bool
end
export GeneralizedPowerCone

function GeneralizedPowerCone{T}(α::Vector{T}, n::Int) where {T <: Real}
    return GeneralizedPowerCone{T}(α, n, false)
end

MOI.dimension(cone::GeneralizedPowerCone) = length(cone.α) + cone.n

function cone_from_moi(::Type{T}, cone::GeneralizedPowerCone{T}) where {T <: Real}
    return Cones.GeneralizedPower{T}(cone.α, cone.n, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.HypoPowerMean`](@ref).

$(TYPEDFIELDS)
"""
struct HypoPowerMeanCone{T <: Real} <: MOI.AbstractVectorSet
    α::Vector{T}
    use_dual::Bool
end
export HypoPowerMeanCone

HypoPowerMeanCone{T}(α::Vector{T}) where {T <: Real} = HypoPowerMeanCone{T}(α, false)

MOI.dimension(cone::HypoPowerMeanCone) = 1 + length(cone.α)

function cone_from_moi(::Type{T}, cone::HypoPowerMeanCone{T}) where {T <: Real}
    return Cones.HypoPowerMean{T}(cone.α, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.HypoGeoMean`](@ref).

$(TYPEDFIELDS)
"""
struct HypoGeoMeanCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
export HypoGeoMeanCone

HypoGeoMeanCone{T}(dim::Int) where {T <: Real} = HypoGeoMeanCone{T}(dim, false)

MOI.dimension(cone::HypoGeoMeanCone) = cone.dim

function cone_from_moi(::Type{T}, cone::HypoGeoMeanCone{T}) where {T <: Real}
    return Cones.HypoGeoMean{T}(cone.dim, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.HypoRootdetTri`](@ref).

$(TYPEDFIELDS)
"""
struct HypoRootdetTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
export HypoRootdetTriCone

function HypoRootdetTriCone{T, R}(dim::Int) where {T <: Real, R <: RealOrComplex{T}}
    return HypoRootdetTriCone{T, R}(dim, false)
end

MOI.dimension(cone::HypoRootdetTriCone where {T <: Real}) = cone.dim

function cone_from_moi(
    ::Type{T},
    cone::HypoRootdetTriCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return Cones.HypoRootdetTri{T, R}(cone.dim, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.HypoPerLog`](@ref).

$(TYPEDFIELDS)
"""
struct HypoPerLogCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
export HypoPerLogCone

HypoPerLogCone{T}(dim::Int) where {T <: Real} = HypoPerLogCone{T}(dim, false)

MOI.dimension(cone::HypoPerLogCone) = cone.dim

function cone_from_moi(::Type{T}, cone::HypoPerLogCone{T}) where {T <: Real}
    return Cones.HypoPerLog{T}(cone.dim, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.HypoPerLogdetTri`](@ref).

$(TYPEDFIELDS)
"""
struct HypoPerLogdetTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
export HypoPerLogdetTriCone

function HypoPerLogdetTriCone{T, R}(dim::Int) where {T <: Real, R <: RealOrComplex{T}}
    return HypoPerLogdetTriCone{T, R}(dim, false)
end

MOI.dimension(cone::HypoPerLogdetTriCone) = cone.dim

function cone_from_moi(
    ::Type{T},
    cone::HypoPerLogdetTriCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return Cones.HypoPerLogdetTri{T, R}(cone.dim, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.EpiPerSepSpectral`](@ref).

$(TYPEDFIELDS)
"""
struct EpiPerSepSpectralCone{T <: Real} <: MOI.AbstractVectorSet
    h::Cones.SepSpectralFun
    Q::Type{<:Cones.ConeOfSquares{T}}
    d::Int
    use_dual::Bool
end
export EpiPerSepSpectralCone

function EpiPerSepSpectralCone{T}(
    h::Cones.SepSpectralFun,
    Q::Type{<:Cones.ConeOfSquares{T}},
    d::Int,
) where {T <: Real}
    return EpiPerSepSpectralCone{T}(h, Q, d, false)
end

MOI.dimension(cone::EpiPerSepSpectralCone) = 2 + Cones.vector_dim(cone.Q, cone.d)

function cone_from_moi(::Type{T}, cone::EpiPerSepSpectralCone{T}) where {T <: Real}
    return Cones.EpiPerSepSpectral{cone.Q, T}(cone.h, cone.d, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.EpiRelEntropy`](@ref).

$(TYPEDFIELDS)
"""
struct EpiRelEntropyCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
export EpiRelEntropyCone

EpiRelEntropyCone{T}(dim::Int) where {T <: Real} = EpiRelEntropyCone{T}(dim, false)

MOI.dimension(cone::EpiRelEntropyCone) = cone.dim

function cone_from_moi(::Type{T}, cone::EpiRelEntropyCone{T}) where {T <: Real}
    return Cones.EpiRelEntropy{T}(cone.dim, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.EpiTrRelEntropyTri`](@ref).

$(TYPEDFIELDS)
"""
struct EpiTrRelEntropyTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
export EpiTrRelEntropyTriCone

function EpiTrRelEntropyTriCone{T, R}(dim::Int) where {T <: Real, R <: RealOrComplex{T}}
    return EpiTrRelEntropyTriCone{T, R}(dim, false)
end

MOI.dimension(cone::EpiTrRelEntropyTriCone) = cone.dim

function cone_from_moi(
    ::Type{T},
    cone::EpiTrRelEntropyTriCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return Cones.EpiTrRelEntropyTri{T, R}(cone.dim, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.WSOSInterpNonnegative`](@ref).

$(TYPEDFIELDS)
"""
struct WSOSInterpNonnegativeCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    U::Int
    Ps::Vector{Matrix{R}}
    use_dual::Bool
end
export WSOSInterpNonnegativeCone

function WSOSInterpNonnegativeCone{T, R}(
    U::Int,
    Ps::Vector{Matrix{R}},
) where {T <: Real, R <: RealOrComplex{T}}
    return WSOSInterpNonnegativeCone{T, R}(U, Ps, false)
end

MOI.dimension(cone::WSOSInterpNonnegativeCone) = cone.U

function cone_from_moi(
    ::Type{T},
    cone::WSOSInterpNonnegativeCone{T, R},
) where {T <: Real, R <: RealOrComplex{T}}
    return Cones.WSOSInterpNonnegative{T, R}(cone.U, cone.Ps, use_dual = cone.use_dual)
end

function _transformation_to(lag::MultivariateBases.LagrangeBasis, gram, weight)
    return StarAlgebras.coeffs(weight, lag) .* MultivariateBases.transformation_to(gram, lag)
end

function cone_from_moi(
    ::Type{T},
    cone::SumOfSquares.WeightedSOSCone{MOI.PositiveSemidefiniteConeTriangle,<:MultivariateBases.LagrangeBasis},
) where {T<:Real}
    return cone_from_moi(
        T,
        WSOSInterpNonnegativeCone{T,T}(
            length(cone.basis),
            _transformation_to.(Ref(cone.basis), cone.gram_bases, cone.weights),
        ),
    )
end


"""
$(TYPEDEF)

See [`Cones.WSOSInterpPosSemidefTri`](@ref).

$(TYPEDFIELDS)
"""
struct WSOSInterpPosSemidefTriCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    use_dual::Bool
end
export WSOSInterpPosSemidefTriCone

function WSOSInterpPosSemidefTriCone{T}(
    R::Int,
    U::Int,
    Ps::Vector{Matrix{T}},
) where {T <: Real}
    return WSOSInterpPosSemidefTriCone{T}(R, U, Ps, false)
end

MOI.dimension(cone::WSOSInterpPosSemidefTriCone) = cone.U * Cones.svec_length(cone.R)

function cone_from_moi(::Type{T}, cone::WSOSInterpPosSemidefTriCone{T}) where {T <: Real}
    return Cones.WSOSInterpPosSemidefTri{T}(
        cone.R,
        cone.U,
        cone.Ps,
        use_dual = cone.use_dual,
    )
end

"""
$(TYPEDEF)

See [`Cones.WSOSInterpEpiNormOne`](@ref).

$(TYPEDFIELDS)
"""
struct WSOSInterpEpiNormOneCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    use_dual::Bool
end
export WSOSInterpEpiNormOneCone

function WSOSInterpEpiNormOneCone{T}(
    R::Int,
    U::Int,
    Ps::Vector{Matrix{T}},
) where {T <: Real}
    return WSOSInterpEpiNormOneCone{T}(R, U, Ps, false)
end

MOI.dimension(cone::WSOSInterpEpiNormOneCone) = cone.U * cone.R

function cone_from_moi(::Type{T}, cone::WSOSInterpEpiNormOneCone{T}) where {T <: Real}
    return Cones.WSOSInterpEpiNormOne{T}(cone.R, cone.U, cone.Ps, use_dual = cone.use_dual)
end

"""
$(TYPEDEF)

See [`Cones.WSOSInterpEpiNormEucl`](@ref).

$(TYPEDFIELDS)
"""
struct WSOSInterpEpiNormEuclCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    use_dual::Bool
end
export WSOSInterpEpiNormEuclCone

function WSOSInterpEpiNormEuclCone{T}(
    R::Int,
    U::Int,
    Ps::Vector{Matrix{T}},
) where {T <: Real}
    return WSOSInterpEpiNormEuclCone{T}(R, U, Ps, false)
end

MOI.dimension(cone::WSOSInterpEpiNormEuclCone) = cone.U * cone.R

function cone_from_moi(::Type{T}, cone::WSOSInterpEpiNormEuclCone{T}) where {T <: Real}
    return Cones.WSOSInterpEpiNormEucl{T}(cone.R, cone.U, cone.Ps, use_dual = cone.use_dual)
end

# all cones

const HypatiaCones{T <: Real} = Union{
    NonnegativeCone{T},
    PosSemidefTriCone{T, T},
    PosSemidefTriCone{T, Complex{T}},
    DoublyNonnegativeTriCone{T},
    PosSemidefTriSparseCone{<:Cones.PSDSparseImpl, T, T},
    PosSemidefTriSparseCone{<:Cones.PSDSparseImpl, T, Complex{T}},
    LinMatrixIneqCone{T},
    EpiNormInfCone{T, T},
    EpiNormInfCone{T, Complex{T}},
    EpiNormEuclCone{T},
    EpiPerSquareCone{T},
    EpiNormSpectralTriCone{T, T},
    EpiNormSpectralTriCone{T, Complex{T}},
    EpiNormSpectralCone{T, T},
    EpiNormSpectralCone{T, Complex{T}},
    MatrixEpiPerSquareCone{T, T},
    MatrixEpiPerSquareCone{T, Complex{T}},
    GeneralizedPowerCone{T},
    HypoPowerMeanCone{T},
    HypoGeoMeanCone{T},
    HypoRootdetTriCone{T, T},
    HypoRootdetTriCone{T, Complex{T}},
    HypoPerLogCone{T},
    HypoPerLogdetTriCone{T, T},
    HypoPerLogdetTriCone{T, Complex{T}},
    EpiPerSepSpectralCone{T},
    EpiRelEntropyCone{T},
    EpiTrRelEntropyTriCone{T, T},
    EpiTrRelEntropyTriCone{T, Complex{T}},
    WSOSInterpNonnegativeCone{T, T},
    WSOSInterpNonnegativeCone{T, Complex{T}},
    WSOSInterpPosSemidefTriCone{T},
    WSOSInterpEpiNormOneCone{T},
    WSOSInterpEpiNormEuclCone{T},
}

const SupportedCone{T <: Real} = Union{
    HypatiaCones{T},
    MOI.Nonnegatives,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.HermitianPositiveSemidefiniteConeTriangle,
    MOI.NormInfinityCone,
    MOI.NormOneCone,
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    MOI.NormSpectralCone,
    MOI.NormNuclearCone,
    MOI.PowerCone{T},
    MOI.DualPowerCone{T},
    MOI.GeometricMeanCone,
    MOI.RootDetConeTriangle,
    MOI.ExponentialCone,
    MOI.DualExponentialCone,
    MOI.LogDetConeTriangle,
    MOI.RelativeEntropyCone,
    SumOfSquares.WeightedSOSCone{MOI.PositiveSemidefiniteConeTriangle,<:MultivariateBases.LagrangeBasis},
}

Base.copy(cone::HypatiaCones) = cone # maybe should deep copy the cone struct, but this is expensive
