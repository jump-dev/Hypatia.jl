#=
Copyright 2018, Chris Coey and contributors

definitions of conic sets not already defined by MathOptInterface
and functions for converting between Hypatia and MOI cone definitions
=#

cone_from_moi(::Type{<:Real}, s::MOI.AbstractVectorSet) = error("MOI set $s is not recognized")

# MOI predefined cones

cone_from_moi(::Type{T}, s::MOI.NormInfinityCone) where {T <: Real} = Cones.EpiNormInf{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.NormOneCone) where {T <: Real} = Cones.EpiNormInf{T, T}(MOI.dimension(s), true)
cone_from_moi(::Type{T}, s::MOI.SecondOrderCone) where {T <: Real} = Cones.EpiNormEucl{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.RotatedSecondOrderCone) where {T <: Real} = Cones.EpiPerSquare{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.ExponentialCone) where {T <: Real} = Cones.HypoPerLog{T}(3) # TODO EpiPerExp
cone_from_moi(::Type{T}, s::MOI.DualExponentialCone) where {T <: Real} = Cones.HypoPerLog{T}(3, true) # TODO EpiPerExp
cone_from_moi(::Type{T}, s::MOI.PowerCone{T}) where {T <: Real} = Cones.Power{T}(T[s.exponent, 1 - s.exponent], 1)
cone_from_moi(::Type{T}, s::MOI.DualPowerCone{T}) where {T <: Real} = Cones.Power{T}(T[s.exponent, 1 - s.exponent], 1, true)
cone_from_moi(::Type{T}, s::MOI.GeometricMeanCone) where {T <: Real} = (l = MOI.dimension(s) - 1; Cones.HypoGeomean{T}(fill(inv(T(l)), l)))
cone_from_moi(::Type{T}, s::MOI.RelativeEntropyCone) where {T <: Real} = Cones.EpiSumPerEntropy{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.PositiveSemidefiniteConeTriangle) where {T <: Real} = Cones.PosSemidefTri{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.LogDetConeTriangle) where {T <: Real} = Cones.HypoPerLogdetTri{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.RootDetConeTriangle) where {T <: Real} = Cones.HypoRootdetTri{T, T}(MOI.dimension(s))

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
cone_from_moi(::Type{T}, s::NonnegativeCone{T}) where {T <: Real} = Cones.Nonnegative{T}(s.dim)

export EpiNormInfinityCone
struct EpiNormInfinityCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    is_dual::Bool
end
EpiNormInfinityCone{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = EpiNormInfinityCone{T, R}(dim, false)
MOI.dimension(cone::EpiNormInfinityCone) = cone.dim
cone_from_moi(::Type{T}, s::EpiNormInfinityCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.EpiNormInf{T, R}(s.dim, s.is_dual)

export EpiNormEuclCone
struct EpiNormEuclCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
end
MOI.dimension(cone::EpiNormEuclCone) = cone.dim
cone_from_moi(::Type{T}, s::EpiNormEuclCone{T}) where {T <: Real} = Cones.EpiNormEucl{T}(s.dim)

export EpiPerSquareCone
struct EpiPerSquareCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
end
MOI.dimension(cone::EpiPerSquareCone) = cone.dim
cone_from_moi(::Type{T}, s::EpiPerSquareCone{T}) where {T <: Real} = Cones.EpiPerSquare{T}(s.dim)

export PowerCone
struct PowerCone{T <: Real} <: MOI.AbstractVectorSet
    alpha::Vector{T}
    n::Int
    is_dual::Bool
end
PowerCone{T}(alpha::Vector{T}, n::Int) where {T <: Real} = PowerCone{T}(alpha, n, false)
MOI.dimension(cone::PowerCone) = length(cone.alpha) + cone.n
cone_from_moi(::Type{T}, s::PowerCone{T}) where {T <: Real} = Cones.Power{T}(s.alpha, s.n, s.is_dual)

export HypoPerLogCone
struct HypoPerLogCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    is_dual::Bool
end
HypoPerLogCone{T}(dim::Int) where {T <: Real} = HypoPerLogCone{T}(dim, false)
MOI.dimension(cone::HypoPerLogCone) = cone.dim
cone_from_moi(::Type{T}, s::HypoPerLogCone{T}) where {T <: Real} = Cones.HypoPerLog{T}(s.dim, s.is_dual)

export EpiPerExpCone
struct EpiPerExpCone{T <: Real} <: MOI.AbstractVectorSet
    is_dual::Bool
end
EpiPerExpCone{T}() where {T <: Real} = EpiPerExpCone{T}(false)
MOI.dimension(cone::EpiPerExpCone) = 3
cone_from_moi(::Type{T}, s::EpiPerExpCone{T}) where {T <: Real} = Cones.EpiPerExp{T}(s.is_dual)

export EpiSumPerEntropyCone
struct EpiSumPerEntropyCone{T <: Real} <: MOI.AbstractVectorSet
    dim::Int
    is_dual::Bool
end
EpiSumPerEntropyCone{T}(dim::Int) where {T <: Real} = EpiSumPerEntropyCone{T}(dim, false)
MOI.dimension(cone::EpiSumPerEntropyCone) = cone.dim
cone_from_moi(::Type{T}, s::EpiSumPerEntropyCone{T}) where {T <: Real} = Cones.EpiSumPerEntropy{T}(s.dim, s.is_dual)

export HypoGeomeanCone
struct HypoGeomeanCone{T <: Real} <: MOI.AbstractVectorSet
    alpha::Vector{T}
    is_dual::Bool
end
HypoGeomeanCone{T}(alpha::Vector{T}) where {T <: Real} = HypoGeomeanCone{T}(alpha, false)
MOI.dimension(cone::HypoGeomeanCone) = 1 + length(cone.alpha)
cone_from_moi(::Type{T}, s::HypoGeomeanCone{T}) where {T <: Real} = Cones.HypoGeomean{T}(s.alpha, s.is_dual)

export EpiNormSpectralCone
struct EpiNormSpectralCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    n::Int
    m::Int
    is_dual::Bool
end
EpiNormSpectralCone{T, R}(n::Int, m::Int) where {R <: RealOrComplex{T}} where {T <: Real} = EpiNormSpectralCone{T, R}(n, m, false)
MOI.dimension(cone::EpiNormSpectralCone{T, T} where {T <: Real}) = 1 + cone.n * cone.m
MOI.dimension(cone::EpiNormSpectralCone{T, Complex{T}} where {T <: Real}) = 1 + 2 * cone.n * cone.m
cone_from_moi(::Type{T}, s::EpiNormSpectralCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.EpiNormSpectral{T, R}(s.n, s.m, s.is_dual)

export MatrixEpiPerSquareCone
struct MatrixEpiPerSquareCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    n::Int
    m::Int
    is_dual::Bool
end
MatrixEpiPerSquareCone{T, R}(n::Int, m::Int) where {R <: RealOrComplex{T}} where {T <: Real} = MatrixEpiPerSquareCone{T, R}(n, m, false)
MOI.dimension(cone::MatrixEpiPerSquareCone{T, T} where {T <: Real}) = Cones.svec_length(cone.n) + 1 + cone.n * cone.m
MOI.dimension(cone::MatrixEpiPerSquareCone{T, Complex{T}} where {T <: Real}) = cone.n ^ 2 + 1 + 2 * cone.n * cone.m
cone_from_moi(::Type{T}, s::MatrixEpiPerSquareCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.MatrixEpiPerSquare{T, R}(s.n, s.m, s.is_dual)

export LinMatrixIneqCone
struct LinMatrixIneqCone{T <: Real} <: MOI.AbstractVectorSet
    As::Vector
    is_dual::Bool
end
LinMatrixIneqCone{T}(As::Vector) where {T <: Real} = LinMatrixIneqCone{T}(As, false)
MOI.dimension(cone::LinMatrixIneqCone) = length(cone.As)
cone_from_moi(::Type{T}, s::LinMatrixIneqCone{T}) where {T <: Real} = Cones.LinMatrixIneq{T}(s.As, s.is_dual)

export PosSemidefTriCone
struct PosSemidefTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
end
MOI.dimension(cone::PosSemidefTriCone{T, T} where {T <: Real}) = cone.dim
MOI.dimension(cone::PosSemidefTriCone{T, Complex{T}} where {T <: Real}) = cone.dim
cone_from_moi(::Type{T}, s::PosSemidefTriCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.PosSemidefTri{T, R}(s.dim)

export HypoPerLogdetTriCone
struct HypoPerLogdetTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    is_dual::Bool
end
HypoPerLogdetTriCone{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = HypoPerLogdetTriCone{T, R}(dim, false)
MOI.dimension(cone::HypoPerLogdetTriCone) = cone.dim
cone_from_moi(::Type{T}, s::HypoPerLogdetTriCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.HypoPerLogdetTri{T, R}(s.dim, s.is_dual)

export HypoRootdetTriCone
struct HypoRootdetTriCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    dim::Int
    is_dual::Bool
end
HypoRootdetTriCone{T, R}(dim::Int) where {R <: RealOrComplex{T}} where {T <: Real} = HypoRootdetTriCone{T, R}(dim, false)
MOI.dimension(cone::HypoRootdetTriCone where {T <: Real}) = cone.dim
cone_from_moi(::Type{T}, s::HypoRootdetTriCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.HypoRootdetTri{T, R}(s.dim, s.is_dual)

export WSOSInterpNonnegativeCone
struct WSOSInterpNonnegativeCone{T <: Real, R <: RealOrComplex{T}} <: MOI.AbstractVectorSet
    U::Int
    Ps::Vector{Matrix{R}}
    is_dual::Bool
end
WSOSInterpNonnegativeCone{T, R}(U::Int, Ps::Vector{Matrix{R}}) where {R <: RealOrComplex{T}} where {T <: Real} = WSOSInterpNonnegativeCone{T, R}(U, Ps, false)
MOI.dimension(cone::WSOSInterpNonnegativeCone) = cone.U
cone_from_moi(::Type{T}, s::WSOSInterpNonnegativeCone{T, R}) where {R <: RealOrComplex{T}} where {T <: Real} = Cones.WSOSInterpNonnegative{T, R}(s.U, s.Ps, s.is_dual)

export WSOSInterpPosSemidefTriCone
struct WSOSInterpPosSemidefTriCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    is_dual::Bool
end
WSOSInterpPosSemidefTriCone{T}(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpPosSemidefTriCone{T}(R, U, Ps, false)
MOI.dimension(cone::WSOSInterpPosSemidefTriCone) = cone.U * Cones.svec_length(cone.R)
cone_from_moi(::Type{T}, s::WSOSInterpPosSemidefTriCone{T}) where {T <: Real} = Cones.WSOSInterpPosSemidefTri{T}(s.R, s.U, s.Ps, s.is_dual)

export WSOSInterpEpiNormEuclCone
struct WSOSInterpEpiNormEuclCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    is_dual::Bool
end
WSOSInterpEpiNormEuclCone{T}(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpEpiNormEuclCone{T}(R, U, Ps, false)
MOI.dimension(cone::WSOSInterpEpiNormEuclCone) = cone.U * cone.R
cone_from_moi(::Type{T}, s::WSOSInterpEpiNormEuclCone{T}) where {T <: Real} = Cones.WSOSInterpEpiNormEucl{T}(s.R, s.U, s.Ps, s.is_dual)

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
    EpiPerExpCone{T},
    EpiSumPerEntropyCone{T},
    HypoGeomeanCone{T},
    EpiNormSpectralCone{T, T},
    EpiNormSpectralCone{T, Complex{T}},
    MatrixEpiPerSquareCone{T, T},
    MatrixEpiPerSquareCone{T, Complex{T}},
    LinMatrixIneqCone{T},
    PosSemidefTriCone{T, T},
    PosSemidefTriCone{T, Complex{T}},
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
    EpiPerExpCone{T},
    EpiSumPerEntropyCone{T},
    HypoGeomeanCone{T},
    EpiNormSpectralCone{T, T},
    EpiNormSpectralCone{T, Complex{T}},
    MatrixEpiPerSquareCone{T, T},
    MatrixEpiPerSquareCone{T, Complex{T}},
    LinMatrixIneqCone{T},
    PosSemidefTriCone{T, T},
    PosSemidefTriCone{T, Complex{T}},
    HypoPerLogdetTriCone{T, T},
    HypoPerLogdetTriCone{T, Complex{T}},
    HypoRootdetTriCone{T, T},
    HypoRootdetTriCone{T, Complex{T}},
    WSOSInterpNonnegativeCone{T, T},
    WSOSInterpNonnegativeCone{T, Complex{T}},
    WSOSInterpPosSemidefTriCone{T},
    WSOSInterpEpiNormEuclCone{T},
    }
