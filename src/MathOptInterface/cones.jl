#=
Copyright 2018, Chris Coey and contributors

definitions of conic sets not already defined by MathOptInterface
and functions for converting between Hypatia and MOI cone definitions
=#

export WSOSInterpNonnegativeCone

struct WSOSInterpNonnegativeCone{T <: Real} <: MOI.AbstractVectorSet # real case only
    U::Int
    Ps::Vector{Matrix{T}}
    is_dual::Bool
end
WSOSInterpNonnegativeCone(U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpNonnegativeCone{T}(U, Ps, false)
MOI.dimension(cone::WSOSInterpNonnegativeCone) = cone.U

export WSOSInterpPossemidefTriCone

struct WSOSInterpPossemidefTriCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    is_dual::Bool
end
WSOSInterpPossemidefTriCone(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpPossemidefTriCone{T}(R, U, Ps, false)
MOI.dimension(cone::WSOSInterpPossemidefTriCone) = U * div(R * (R + 1), 2)

export WSOSInterpEpiNormEuclCone

struct WSOSInterpEpiNormEuclCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    is_dual::Bool
end
WSOSInterpEpiNormEuclCone(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpEpiNormEuclCone{T}(R, U, Ps, false)
MOI.dimension(cone::WSOSInterpEpiNormEuclCone) = U * R

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
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.LogDetConeTriangle,
    MOI.RootDetConeTriangle,
    WSOSInterpNonnegativeCone{T},
    WSOSInterpPossemidefTriCone{T},
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
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.LogDetConeTriangle,
    MOI.RootDetConeTriangle,
    WSOSInterpNonnegativeCone{T},
    WSOSInterpPossemidefTriCone{T},
    WSOSInterpEpiNormEuclCone{T},
    }

# MOI cones for which no transformation is needed
cone_from_moi(::Type{<:Real}, s::MOI.AbstractVectorSet) = error("MOI set $s is not recognized")
cone_from_moi(::Type{T}, s::MOI.NormInfinityCone) where {T <: Real} = Cones.EpiNormInf{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.NormOneCone) where {T <: Real} = Cones.EpiNormInf{T, T}(MOI.dimension(s), true)
cone_from_moi(::Type{T}, s::MOI.SecondOrderCone) where {T <: Real} = Cones.EpiNormEucl{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.RotatedSecondOrderCone) where {T <: Real} = Cones.EpiPerSquare{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.ExponentialCone) where {T <: Real} = Cones.HypoPerLog{T}(3) # TODO EpiPerExp
cone_from_moi(::Type{T}, s::MOI.DualExponentialCone) where {T <: Real} = Cones.HypoPerLog{T}(3, true) # TODO EpiPerExp
cone_from_moi(::Type{T}, s::MOI.PowerCone{T}) where {T <: Real} = Cones.Power{T}([s.exponent, 1 - s.exponent], 1)
cone_from_moi(::Type{T}, s::MOI.DualPowerCone{T}) where {T <: Real} = Cones.Power{T}([s.exponent, 1 - s.exponent], 1, true)
cone_from_moi(::Type{T}, s::MOI.GeometricMeanCone) where {T <: Real} = (l = MOI.dimension(s) - 1; Cones.HypoGeomean{T}(fill(inv(l), l)))
cone_from_moi(::Type{T}, s::MOI.PositiveSemidefiniteConeTriangle) where {T <: Real} = Cones.PosSemidefTri{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.LogDetConeTriangle) where {T <: Real} = Cones.HypoPerLogdetTri{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.RootDetConeTriangle) where {T <: Real} = Cones.HypoRootdetTri{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::WSOSInterpNonnegativeCone{T}) where {T <: Real} = Cones.WSOSInterpNonnegative{T, T}(s.U, s.Ps, s.is_dual)
cone_from_moi(::Type{T}, s::WSOSInterpPossemidefTriCone{T}) where {T <: Real} = Cones.WSOSInterpPosSemidefTri{T}(s.R, s.U, s.Ps, s.is_dual)
cone_from_moi(::Type{T}, s::WSOSInterpEpiNormEuclCone{T}) where {T <: Real} = Cones.WSOSInterpEpiNormEucl{T}(s.R, s.U, s.Ps, s.is_dual)

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

# TODO scale for WSOSInterpPossemidefTriCone?
