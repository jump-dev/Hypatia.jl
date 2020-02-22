#=
Copyright 2018, Chris Coey and contributors

definitions of conic sets not already defined by MathOptInterface
and functions for converting between Hypatia and MOI cone definitions
=#

cone_from_moi(::Type{<:Real}, cone::MOI.AbstractVectorSet) = error("MOI set $s is not recognized")

# MOI predefined cones

cone_from_moi(::Type{T}, cone::MOI.NormInfinityCone) where {T <: Real} = Cones.EpiNormInf{T, T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.NormOneCone) where {T <: Real} = Cones.EpiNormInf{T, T}(MOI.dimension(cone), use_dual = true)
cone_from_moi(::Type{T}, cone::MOI.SecondOrderCone) where {T <: Real} = Cones.EpiNormEucl{T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.RotatedSecondOrderCone) where {T <: Real} = Cones.EpiPerSquare{T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.ExponentialCone) where {T <: Real} = Cones.HypoPerLog{T}(3)
cone_from_moi(::Type{T}, cone::MOI.DualExponentialCone) where {T <: Real} = Cones.HypoPerLog{T}(3, use_dual = true)
cone_from_moi(::Type{T}, cone::MOI.PowerCone{T}) where {T <: Real} = Cones.Power{T}(T[cone.exponent, 1 - cone.exponent], 1)
cone_from_moi(::Type{T}, cone::MOI.DualPowerCone{T}) where {T <: Real} = Cones.Power{T}(T[cone.exponent, 1 - cone.exponent], 1, use_dual = true)
cone_from_moi(::Type{T}, cone::MOI.GeometricMeanCone) where {T <: Real} = (l = MOI.dimension(cone) - 1; Cones.HypoGeomean{T}(fill(inv(T(l)), l)))
cone_from_moi(::Type{T}, cone::MOI.RelativeEntropyCone) where {T <: Real} = Cones.EpiSumPerEntropy{T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.NormSpectralCone) where {T <: Real} = Cones.EpiNormSpectral{T, T}(cone.row_dim, cone.column_dim)
cone_from_moi(::Type{T}, cone::MOI.NormNuclearCone) where {T <: Real} = Cones.EpiNormSpectral{T, T}(cone.row_dim, cone.column_dim, use_dual = true)
cone_from_moi(::Type{T}, cone::MOI.PositiveSemidefiniteConeTriangle) where {T <: Real} = Cones.PosSemidefTri{T, T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.LogDetConeTriangle) where {T <: Real} = Cones.HypoPerLogdetTri{T, T}(MOI.dimension(cone))
cone_from_moi(::Type{T}, cone::MOI.RootDetConeTriangle) where {T <: Real} = Cones.HypoRootdetTri{T, T}(MOI.dimension(cone))

# transformations for MOI cones not in svec (scaled lower triangle) form

SvecCone = Union{Cones.PosSemidefTri, Cones.HypoPerLogdetTri, Cones.HypoRootdetTri}
svec_offset(cone::Cones.PosSemidefTri) = 0
svec_offset(cone::Cones.HypoPerLogdetTri) = 2
svec_offset(cone::Cones.HypoRootdetTri) = 1

untransform_cone_vec(cone::Cones.Cone, vec::AbstractVector) = nothing
untransform_cone_vec(cone::SvecCone, vec::AbstractVector) = (@views ModelUtilities.svec_to_vec!(vec[(1 + svec_offset(cone)):end]))

const rt2 = sqrt(2)
svec_scale(dim) = [(i == j ? 1.0 : rt2) for i in 1:round(Int, sqrt(0.25 + 2 * dim) - 0.5) for j in 1:i]

get_affine_data_vov(cone::Cones.Cone, dim::Int) = fill(-1.0, dim)
function get_affine_data_vov(cone::SvecCone, dim::Int)
    offset = svec_offset(cone)
    VGi = vcat(ones(offset), svec_scale(dim - offset))
    VGi .*= -1
    return VGi
end

get_affine_data_vaf(cone::Cones.Cone, fi::MOI.VectorAffineFunction{Float64}, dim::Int) = ([-vt.scalar_term.coefficient for vt in fi.terms], fi.constants)
function get_affine_data_vaf(cone::SvecCone, fi::MOI.VectorAffineFunction{Float64}, dim::Int)
    offset = svec_offset(cone)
    scalevec = vcat(ones(offset), svec_scale(dim - offset))
    VGi = [-vt.scalar_term.coefficient * scalevec[vt.output_index] for vt in fi.terms]
    Vhi = scalevec .* fi.constants
    return (VGi, Vhi)
end

# Hypatia predefined cones
# NOTE some are equivalent to above MOI predefined cones, but we define again for the sake of consistency

export NonnegativeCone
struct NonnegativeCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
end
MOI.dimension(cone::NonnegativeCone) = cone.dim
cone_from_moi(::Type{T}, cone::NonnegativeCone{T}) where {T <: Real} = Cones.Nonnegative{T}(cone.dim)

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

export PowerCone
struct PowerCone{T <: Real} <: MOI.AbstractVectorSet
    alpha::Vector{T}
    n::Int
    use_dual::Bool
end
PowerCone{T}(alpha::Vector{T}, n::Int) where {T <: Real} = PowerCone{T}(alpha, n, false)
MOI.dimension(cone::PowerCone) = length(cone.alpha) + cone.n
cone_from_moi(::Type{T}, cone::PowerCone{T}) where {T <: Real} = Cones.Power{T}(cone.alpha, cone.n, use_dual = cone.use_dual)

export HypoPerLogCone
struct HypoPerLogCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
HypoPerLogCone{T}(dim::Int) where {T <: Real} = HypoPerLogCone{T}(dim, false)
MOI.dimension(cone::HypoPerLogCone) = cone.dim
cone_from_moi(::Type{T}, cone::HypoPerLogCone{T}) where {T <: Real} = Cones.HypoPerLog{T}(cone.dim, use_dual = cone.use_dual)

export EpiSumPerEntropyCone
struct EpiSumPerEntropyCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
EpiSumPerEntropyCone{T}(dim::Int) where {T <: Real} = EpiSumPerEntropyCone{T}(dim, false)
MOI.dimension(cone::EpiSumPerEntropyCone) = cone.dim
cone_from_moi(::Type{T}, cone::EpiSumPerEntropyCone{T}) where {T <: Real} = Cones.EpiSumPerEntropy{T}(cone.dim, use_dual = cone.use_dual)

export HypoGeomeanCone
struct HypoGeomeanCone{T <: Real} <: MOI.AbstractVectorSet
    alpha::Vector{T}
    use_dual::Bool
end
HypoGeomeanCone{T}(alpha::Vector{T}) where {T <: Real} = HypoGeomeanCone{T}(alpha, false)
MOI.dimension(cone::HypoGeomeanCone) = 1 + length(cone.alpha)
cone_from_moi(::Type{T}, cone::HypoGeomeanCone{T}) where {T <: Real} = Cones.HypoGeomean{T}(cone.alpha, use_dual = cone.use_dual)

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

export LinMatrixIneqCone
struct LinMatrixIneqCone{T <: Real} <: MOI.AbstractVectorSet
    As::Vector
    use_dual::Bool
end
LinMatrixIneqCone{T}(As::Vector) where {T <: Real} = LinMatrixIneqCone{T}(As, false)
MOI.dimension(cone::LinMatrixIneqCone) = length(cone.As)
cone_from_moi(::Type{T}, cone::LinMatrixIneqCone{T}) where {T <: Real} = Cones.LinMatrixIneq{T}(cone.As, use_dual = cone.use_dual)

export PosSemidefTriCone
struct PosSemidefTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
end
MOI.dimension(cone::PosSemidefTriCone) = cone.dim
cone_from_moi(::Type{T}, cone::PosSemidefTriCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.PosSemidefTri{T, R}(cone.dim)

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

export HypoPerLogdetTriCone
struct HypoPerLogdetTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
HypoPerLogdetTriCone{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = HypoPerLogdetTriCone{T, R}(dim, false)
MOI.dimension(cone::HypoPerLogdetTriCone) = cone.dim
cone_from_moi(::Type{T}, cone::HypoPerLogdetTriCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.HypoPerLogdetTri{T, R}(cone.dim, use_dual = cone.use_dual)

export HypoRootdetTriCone
struct HypoRootdetTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    use_dual::Bool
end
HypoRootdetTriCone{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = HypoRootdetTriCone{T, R}(dim, false)
MOI.dimension(cone::HypoRootdetTriCone where {T <: Real}) = cone.dim
cone_from_moi(::Type{T}, cone::HypoRootdetTriCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.HypoRootdetTri{T, R}(cone.dim, use_dual = cone.use_dual)

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
# TODO avoid repeating the tuple and union below

const MOIOtherConesList(::Type{T}) where {T <: Real} = (
    MOI.NormInfinityCone,
    MOI.NormOneCone,
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    MOI.ExponentialCone,
    MOI.DualExponentialCone,
    MOI.PowerCone{T},
    MOI.DualPowerCone{T},
    MOI.GeometricMeanCone,
    MOI.RelativeEntropyCone,
    MOI.NormSpectralCone,
    MOI.NormNuclearCone,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.LogDetConeTriangle,
    MOI.RootDetConeTriangle,
    NonnegativeCone{T},
    EpiNormInfinityCone{T, T},
    EpiNormInfinityCone{T, Complex{T}},
    EpiNormEuclCone{T},
    EpiPerSquareCone{T},
    PowerCone{T},
    HypoPerLogCone{T},
    EpiSumPerEntropyCone{T},
    HypoGeomeanCone{T},
    EpiNormSpectralCone{T, T},
    EpiNormSpectralCone{T, Complex{T}},
    MatrixEpiPerSquareCone{T, T},
    MatrixEpiPerSquareCone{T, Complex{T}},
    LinMatrixIneqCone{T},
    PosSemidefTriCone{T, T},
    PosSemidefTriCone{T, Complex{T}},
    PosSemidefTriSparseCone{T, T},
    PosSemidefTriSparseCone{T, Complex{T}},
    HypoPerLogdetTriCone{T, T},
    HypoPerLogdetTriCone{T, Complex{T}},
    HypoRootdetTriCone{T, T},
    HypoRootdetTriCone{T, Complex{T}},
    WSOSInterpNonnegativeCone{T, T},
    WSOSInterpNonnegativeCone{T, Complex{T}},
    WSOSInterpPosSemidefTriCone{T},
    WSOSInterpEpiNormEuclCone{T},
    )

const MOIOtherCones{T <: Real} = Union{
    MOI.NormInfinityCone,
    MOI.NormOneCone,
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    MOI.ExponentialCone,
    MOI.DualExponentialCone,
    MOI.PowerCone{T},
    MOI.DualPowerCone{T},
    MOI.GeometricMeanCone,
    MOI.RelativeEntropyCone,
    MOI.NormSpectralCone,
    MOI.NormNuclearCone,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.LogDetConeTriangle,
    MOI.RootDetConeTriangle,
    NonnegativeCone{T},
    EpiNormInfinityCone{T, T},
    EpiNormInfinityCone{T, Complex{T}},
    EpiNormEuclCone{T},
    EpiPerSquareCone{T},
    PowerCone{T},
    HypoPerLogCone{T},
    EpiSumPerEntropyCone{T},
    HypoGeomeanCone{T},
    EpiNormSpectralCone{T, T},
    EpiNormSpectralCone{T, Complex{T}},
    MatrixEpiPerSquareCone{T, T},
    MatrixEpiPerSquareCone{T, Complex{T}},
    LinMatrixIneqCone{T},
    PosSemidefTriCone{T, T},
    PosSemidefTriCone{T, Complex{T}},
    PosSemidefTriSparseCone{T, T},
    PosSemidefTriSparseCone{T, Complex{T}},
    HypoPerLogdetTriCone{T, T},
    HypoPerLogdetTriCone{T, Complex{T}},
    HypoRootdetTriCone{T, T},
    HypoRootdetTriCone{T, Complex{T}},
    WSOSInterpNonnegativeCone{T, T},
    WSOSInterpNonnegativeCone{T, Complex{T}},
    WSOSInterpPosSemidefTriCone{T},
    WSOSInterpEpiNormEuclCone{T},
    }
