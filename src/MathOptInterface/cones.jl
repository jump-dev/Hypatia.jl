#=
definitions of conic sets not already defined by MathOptInterface
and functions for converting between Hypatia and MOI cone definitions
=#

cone_from_moi(::Type{<:Real}, cone::MOI.AbstractVectorSet) = error("MOI set $cone is not recognized")

# MOI predefined cones

cone_from_moi(::Type{T}, cone::MOI.PositiveSemidefiniteConeTriangle) where {T <: Real} = Cones.PosSemidefTri{T, T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.NormInfinityCone) where {T <: Real} = Cones.EpiNormInf{T, T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.NormOneCone) where {T <: Real} = Cones.EpiNormInf{T, T}(MOI.dimension(cone), use_dual = true)
cone_from_moi(::Type{T}, cone::MOI.SecondOrderCone) where {T <: Real} = Cones.EpiNormEucl{T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.RotatedSecondOrderCone) where {T <: Real} = Cones.EpiPerSquare{T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.NormSpectralCone) where {T <: Real} = Cones.EpiNormSpectral{T, T}(extrema((cone.row_dim, cone.column_dim))...)
cone_from_moi(::Type{T}, cone::MOI.NormNuclearCone) where {T <: Real} = Cones.EpiNormSpectral{T, T}(extrema((cone.row_dim, cone.column_dim))..., use_dual = true)
cone_from_moi(::Type{T}, cone::MOI.PowerCone{T}) where {T <: Real} = Cones.GeneralizedPower{T}(T[cone.exponent, 1 - cone.exponent], 1)
cone_from_moi(::Type{T}, cone::MOI.DualPowerCone{T}) where {T <: Real} = Cones.GeneralizedPower{T}(T[cone.exponent, 1 - cone.exponent], 1, use_dual = true)
cone_from_moi(::Type{T}, cone::MOI.GeometricMeanCone) where {T <: Real} = (l = MOI.dimension(cone) - 1; Cones.HypoGeoMean{T}(1 + l))
cone_from_moi(::Type{T}, cone::MOI.RootDetConeTriangle) where {T <: Real} = Cones.HypoRootdetTri{T, T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.ExponentialCone) where {T <: Real} = Cones.HypoPerLog{T}(3)
cone_from_moi(::Type{T}, cone::MOI.DualExponentialCone) where {T <: Real} = Cones.HypoPerLog{T}(3, use_dual = true)
cone_from_moi(::Type{T}, cone::MOI.LogDetConeTriangle) where {T <: Real} = Cones.HypoPerLogdetTri{T, T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.RelativeEntropyCone) where {T <: Real} = Cones.EpiRelEntropy{T}(MOI.dimension(cone))

# transformations fallbacks
needs_untransform(::MOI.AbstractVectorSet) = false
untransform_affine(::MOI.AbstractVectorSet, vals::AbstractVector) = nothing
permute_affine(::MOI.AbstractVectorSet, idxs::AbstractVector) = idxs
rescale_affine(::MOI.AbstractVectorSet, vals::AbstractVector) = vals
rescale_affine(::MOI.AbstractVectorSet, vals::AbstractVector, ::AbstractVector) = vals

# transformations (transposition of matrix) for MOI rectangular matrix cones with matrix of more rows than columns

NonSquareMatrixCone = Union{MOI.NormSpectralCone, MOI.NormNuclearCone}

needs_untransform(cone::NonSquareMatrixCone) = (cone.row_dim > cone.column_dim)

function untransform_affine(cone::NonSquareMatrixCone, vals::AbstractVector)
    vals[2:end] = reshape(vals[2:end], cone.column_dim, cone.row_dim)'
    return vals
end

function permute_affine(cone::NonSquareMatrixCone, idxs::AbstractVector)
    if cone.row_dim > cone.column_dim
        idxs_new = collect(idxs)
        # transpose the matrix part
        for k in 2:length(idxs)
            (col_old, row_old) = divrem(idxs[k] - 2, cone.row_dim)
            idxs_new[k] = idxs[2 + row_old * cone.column_dim + col_old]
        end
        return idxs_new
    end
    return idxs
end

# transformations (svec rescaling) for MOI symmetric matrix cones not in svec (scaled lower triangle) form
const SvecCone = Union{MOI.PositiveSemidefiniteConeTriangle, MOI.LogDetConeTriangle, MOI.RootDetConeTriangle}

svec_offset(::MOI.PositiveSemidefiniteConeTriangle) = 1
svec_offset(::MOI.RootDetConeTriangle) = 2
svec_offset(::MOI.LogDetConeTriangle) = 3

needs_untransform(::SvecCone) = true

function untransform_affine(cone::SvecCone, vals::AbstractVector)
    @views svec_vals = vals[svec_offset(cone):end]
    ModelUtilities.svec_to_vec!(svec_vals)
    return vals
end

function rescale_affine(cone::SvecCone, vals::AbstractVector)
    vals = collect(vals)
    @views svec_vals = vals[svec_offset(cone):end]
    ModelUtilities.vec_to_svec!(svec_vals)
    return vals
end

function rescale_affine(cone::SvecCone, vals::AbstractVector, idxs::AbstractVector)
    scal_start = svec_offset(cone) - 1
    rt2 = sqrt(eltype(vals)(2))
    for i in eachindex(vals)
        shifted_idx = idxs[i] - scal_start
        if shifted_idx > 0 && !MOI.Utilities.is_diagonal_vectorized_index(shifted_idx)
            vals[i] *= rt2
        end
    end
    return vals
end

# Hypatia predefined cones
# NOTE some are equivalent to above MOI predefined cones, but we define again for the sake of consistency

export NonnegativeCone
struct NonnegativeCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
end
MOI.dimension(cone::NonnegativeCone) = cone.dim
cone_from_moi(::Type{T}, cone::NonnegativeCone{T}) where {T <: Real} = Cones.Nonnegative{T}(cone.dim)

export PosSemidefTriCone
struct PosSemidefTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
end
MOI.dimension(cone::PosSemidefTriCone) = cone.dim
cone_from_moi(::Type{T}, cone::PosSemidefTriCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.PosSemidefTri{T, R}(cone.dim)

export DoublyNonnegativeTriCone
struct DoublyNonnegativeTriCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
DoublyNonnegativeTriCone{T}(dim::Int) where {T <: Real} = DoublyNonnegativeTriCone{T}(dim, false)
MOI.dimension(cone::DoublyNonnegativeTriCone where {T <: Real}) = cone.dim
cone_from_moi(::Type{T}, cone::DoublyNonnegativeTriCone{T}) where {T <: Real} = Cones.DoublyNonnegativeTri{T}(cone.dim, use_dual = cone.use_dual)

export PosSemidefTriSparseCone
struct PosSemidefTriSparseCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    side::Int
    row_idxs::Vector{Int}
    col_idxs::Vector{Int}
    use_dual::Bool
end
PosSemidefTriSparseCone{T, R}(side::Int, row_idxs::Vector{Int}, col_idxs::Vector{Int}) where {R <: RealOrComplex{T}} where {T <: Real} = PosSemidefTriSparseCone{T, R}(side, row_idxs, col_idxs, false)
MOI.dimension(cone::PosSemidefTriSparseCone{T, T} where {T <: Real}) = length(cone.row_idxs)
MOI.dimension(cone::PosSemidefTriSparseCone{T, Complex{T}} where {T <: Real}) = (2 * length(cone.row_idxs) - cone.side)
cone_from_moi(::Type{T}, cone::PosSemidefTriSparseCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.PosSemidefTriSparse{T, R}(cone.side, cone.row_idxs, cone.col_idxs, use_dual = cone.use_dual)

export LinMatrixIneqCone
struct LinMatrixIneqCone{T <: Real} <: MOI.AbstractVectorSet
    As::Vector
    use_dual::Bool
end
LinMatrixIneqCone{T}(As::Vector) where {T <: Real} = LinMatrixIneqCone{T}(As, false)
MOI.dimension(cone::LinMatrixIneqCone) = length(cone.As)
cone_from_moi(::Type{T}, cone::LinMatrixIneqCone{T}) where {T <: Real} = Cones.LinMatrixIneq{T}(cone.As, use_dual = cone.use_dual)

export EpiNormInfinityCone
struct EpiNormInfinityCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
EpiNormInfinityCone{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = EpiNormInfinityCone{T, R}(dim, false)
MOI.dimension(cone::EpiNormInfinityCone) = cone.dim
cone_from_moi(::Type{T}, cone::EpiNormInfinityCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.EpiNormInf{T, R}(cone.dim, use_dual = cone.use_dual)

export EpiNormEuclCone
struct EpiNormEuclCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
end
MOI.dimension(cone::EpiNormEuclCone) = cone.dim
cone_from_moi(::Type{T}, cone::EpiNormEuclCone{T}) where {T <: Real} = Cones.EpiNormEucl{T}(cone.dim)

export EpiPerSquareCone
struct EpiPerSquareCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
end
MOI.dimension(cone::EpiPerSquareCone) = cone.dim
cone_from_moi(::Type{T}, cone::EpiPerSquareCone{T}) where {T <: Real} = Cones.EpiPerSquare{T}(cone.dim)

export EpiNormSpectralCone
struct EpiNormSpectralCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    n::Int
    m::Int
    use_dual::Bool
end
EpiNormSpectralCone{T, R}(n::Int, m::Int) where {R <: RealOrComplex{T}} where {T <: Real} = EpiNormSpectralCone{T, R}(n, m, false)
MOI.dimension(cone::EpiNormSpectralCone{T, T} where {T <: Real}) = 1 + cone.n * cone.m
MOI.dimension(cone::EpiNormSpectralCone{T, Complex{T}} where {T <: Real}) = 1 + 2 * cone.n * cone.m
cone_from_moi(::Type{T}, cone::EpiNormSpectralCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.EpiNormSpectral{T, R}(cone.n, cone.m, use_dual = cone.use_dual)

export MatrixEpiPerSquareCone
struct MatrixEpiPerSquareCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    n::Int
    m::Int
    use_dual::Bool
end
MatrixEpiPerSquareCone{T, R}(n::Int, m::Int) where {R <: RealOrComplex{T}} where {T <: Real} = MatrixEpiPerSquareCone{T, R}(n, m, false)
MOI.dimension(cone::MatrixEpiPerSquareCone{T, T} where {T <: Real}) = Cones.svec_length(cone.n) + 1 + cone.n * cone.m
MOI.dimension(cone::MatrixEpiPerSquareCone{T, Complex{T}} where {T <: Real}) = cone.n ^ 2 + 1 + 2 * cone.n * cone.m
cone_from_moi(::Type{T}, cone::MatrixEpiPerSquareCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.MatrixEpiPerSquare{T, R}(cone.n, cone.m, use_dual = cone.use_dual)

export GeneralizedPowerCone
struct GeneralizedPowerCone{T <: Real} <: MOI.AbstractVectorSet
    alpha::Vector{T}
    n::Int
    use_dual::Bool
end
GeneralizedPowerCone{T}(alpha::Vector{T}, n::Int) where {T <: Real} = GeneralizedPowerCone{T}(alpha, n, false)
MOI.dimension(cone::GeneralizedPowerCone) = length(cone.alpha) + cone.n
cone_from_moi(::Type{T}, cone::GeneralizedPowerCone{T}) where {T <: Real} = Cones.GeneralizedPower{T}(cone.alpha, cone.n, use_dual = cone.use_dual)

export HypoPowerMeanCone
struct HypoPowerMeanCone{T <: Real} <: MOI.AbstractVectorSet
    alpha::Vector{T}
    use_dual::Bool
end
HypoPowerMeanCone{T}(alpha::Vector{T}) where {T <: Real} = HypoPowerMeanCone{T}(alpha, false)
MOI.dimension(cone::HypoPowerMeanCone) = 1 + length(cone.alpha)
cone_from_moi(::Type{T}, cone::HypoPowerMeanCone{T}) where {T <: Real} = Cones.HypoPowerMean{T}(cone.alpha, use_dual = cone.use_dual)

export HypoGeoMeanCone
struct HypoGeoMeanCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
HypoGeoMeanCone{T}(dim::Int) where {T <: Real} = HypoGeoMeanCone{T}(dim, false)
MOI.dimension(cone::HypoGeoMeanCone) = cone.dim
cone_from_moi(::Type{T}, cone::HypoGeoMeanCone{T}) where {T <: Real} = Cones.HypoGeoMean{T}(cone.dim, use_dual = cone.use_dual)

export HypoRootdetTriCone
struct HypoRootdetTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
HypoRootdetTriCone{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = HypoRootdetTriCone{T, R}(dim, false)
MOI.dimension(cone::HypoRootdetTriCone where {T <: Real}) = cone.dim
cone_from_moi(::Type{T}, cone::HypoRootdetTriCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.HypoRootdetTri{T, R}(cone.dim, use_dual = cone.use_dual)

export HypoPerLogCone
struct HypoPerLogCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
HypoPerLogCone{T}(dim::Int) where {T <: Real} = HypoPerLogCone{T}(dim, false)
MOI.dimension(cone::HypoPerLogCone) = cone.dim
cone_from_moi(::Type{T}, cone::HypoPerLogCone{T}) where {T <: Real} = Cones.HypoPerLog{T}(cone.dim, use_dual = cone.use_dual)

export HypoPerLogdetTriCone
struct HypoPerLogdetTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
HypoPerLogdetTriCone{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = HypoPerLogdetTriCone{T, R}(dim, false)
MOI.dimension(cone::HypoPerLogdetTriCone) = cone.dim
cone_from_moi(::Type{T}, cone::HypoPerLogdetTriCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.HypoPerLogdetTri{T, R}(cone.dim, use_dual = cone.use_dual)

export EpiPerEntropyCone
struct EpiPerEntropyCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
EpiPerEntropyCone{T}(dim::Int) where {T <: Real} = EpiPerEntropyCone{T}(dim, false)
MOI.dimension(cone::EpiPerEntropyCone) = cone.dim
cone_from_moi(::Type{T}, cone::EpiPerEntropyCone{T}) where {T <: Real} = Cones.EpiPerEntropy{T}(cone.dim, use_dual = cone.use_dual)

export EpiPerTraceEntropyTriCone
struct EpiPerTraceEntropyTriCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
EpiPerTraceEntropyTriCone{T}(dim::Int) where {T <: Real} = EpiPerTraceEntropyTriCone{T}(dim, false)
MOI.dimension(cone::EpiPerTraceEntropyTriCone where {T <: Real}) = cone.dim
cone_from_moi(::Type{T}, cone::EpiPerTraceEntropyTriCone{T}) where {T <: Real} = Cones.EpiPerTraceEntropyTri{T}(cone.dim, use_dual = cone.use_dual)

export EpiRelEntropyCone
struct EpiRelEntropyCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
EpiRelEntropyCone{T}(dim::Int) where {T <: Real} = EpiRelEntropyCone{T}(dim, false)
MOI.dimension(cone::EpiRelEntropyCone) = cone.dim
cone_from_moi(::Type{T}, cone::EpiRelEntropyCone{T}) where {T <: Real} = Cones.EpiRelEntropy{T}(cone.dim, use_dual = cone.use_dual)

export EpiTraceRelEntropyTriCone
struct EpiTraceRelEntropyTriCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
EpiTraceRelEntropyTriCone{T}(dim::Int) where {T <: Real} = EpiTraceRelEntropyTriCone{T}(dim, false)
MOI.dimension(cone::EpiTraceRelEntropyTriCone where {T <: Real}) = cone.dim
cone_from_moi(::Type{T}, cone::EpiTraceRelEntropyTriCone{T}) where {T <: Real} = Cones.EpiTraceRelEntropyTri{T}(cone.dim, use_dual = cone.use_dual)

export WSOSInterpNonnegativeCone
struct WSOSInterpNonnegativeCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    U::Int
    Ps::Vector{Matrix{R}}
    use_dual::Bool
end
WSOSInterpNonnegativeCone{T, R}(U::Int, Ps::Vector{Matrix{R}}) where {R <: RealOrComplex{T}} where {T <: Real} = WSOSInterpNonnegativeCone{T, R}(U, Ps, false)
MOI.dimension(cone::WSOSInterpNonnegativeCone) = cone.U
cone_from_moi(::Type{T}, cone::WSOSInterpNonnegativeCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.WSOSInterpNonnegative{T, R}(cone.U, cone.Ps, use_dual = cone.use_dual)

export WSOSInterpPosSemidefTriCone
struct WSOSInterpPosSemidefTriCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    use_dual::Bool
end
WSOSInterpPosSemidefTriCone{T}(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpPosSemidefTriCone{T}(R, U, Ps, false)
MOI.dimension(cone::WSOSInterpPosSemidefTriCone) = cone.U * Cones.svec_length(cone.R)
cone_from_moi(::Type{T}, cone::WSOSInterpPosSemidefTriCone{T}) where {T <: Real} = Cones.WSOSInterpPosSemidefTri{T}(cone.R, cone.U, cone.Ps, use_dual = cone.use_dual)

export WSOSInterpEpiNormOneCone
struct WSOSInterpEpiNormOneCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    use_dual::Bool
end
WSOSInterpEpiNormOneCone{T}(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpEpiNormOneCone{T}(R, U, Ps, false)
MOI.dimension(cone::WSOSInterpEpiNormOneCone) = cone.U * cone.R
cone_from_moi(::Type{T}, cone::WSOSInterpEpiNormOneCone{T}) where {T <: Real} = Cones.WSOSInterpEpiNormOne{T}(cone.R, cone.U, cone.Ps, use_dual = cone.use_dual)

export WSOSInterpEpiNormEuclCone
struct WSOSInterpEpiNormEuclCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    use_dual::Bool
end
WSOSInterpEpiNormEuclCone{T}(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpEpiNormEuclCone{T}(R, U, Ps, false)
MOI.dimension(cone::WSOSInterpEpiNormEuclCone) = cone.U * cone.R
cone_from_moi(::Type{T}, cone::WSOSInterpEpiNormEuclCone{T}) where {T <: Real} = Cones.WSOSInterpEpiNormEucl{T}(cone.R, cone.U, cone.Ps, use_dual = cone.use_dual)

# all cones

const HypatiaCones{T <: Real} = Union{
    NonnegativeCone{T},
    PosSemidefTriCone{T, T},
    PosSemidefTriCone{T, Complex{T}},
    DoublyNonnegativeTriCone{T},
    PosSemidefTriSparseCone{T, T},
    PosSemidefTriSparseCone{T, Complex{T}},
    LinMatrixIneqCone{T},
    EpiNormInfinityCone{T, T},
    EpiNormInfinityCone{T, Complex{T}},
    EpiNormEuclCone{T},
    EpiPerSquareCone{T},
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
    EpiPerEntropyCone{T},
    EpiPerTraceEntropyTriCone{T},
    EpiRelEntropyCone{T},
    EpiTraceRelEntropyTriCone{T},
    WSOSInterpNonnegativeCone{T, T},
    WSOSInterpNonnegativeCone{T, Complex{T}},
    WSOSInterpPosSemidefTriCone{T},
    WSOSInterpEpiNormOneCone{T},
    WSOSInterpEpiNormEuclCone{T},
    }

const SupportedCones{T <: Real} = Union{
    HypatiaCones{T},
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.Nonpositives,
    MOI.PositiveSemidefiniteConeTriangle,
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
    }

Base.copy(cone::HypatiaCones) = cone # NOTE maybe should deep copy the cone struct, but this is expensive
