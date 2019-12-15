#=
Copyright 2018, Chris Coey and contributors

definitions of conic sets not already defined by MathOptInterface
and functions for converting between Hypatia and MOI cone definitions
=#

export WSOSPolyInterpCone

struct WSOSPolyInterpCone{T <: Real} <: MOI.AbstractVectorSet # real case only
    dimension::Int
    Ps::Vector{Matrix{T}}
    is_dual::Bool
end
WSOSPolyInterpCone(dimension::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSPolyInterpCone{T}(dimension, Ps, false)

export WSOSPolyInterpMatCone # TODO rename

struct WSOSPolyInterpMatCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    ipwt::Vector{Matrix{T}}
    is_dual::Bool
end
WSOSPolyInterpMatCone(R::Int, U::Int, ipwt::Vector{Matrix{T}}) where {T <: Real} = WSOSPolyInterpMatCone{T}(R, U, ipwt, false)

export WSOSPolyInterpSOCCone # TODO rename

struct WSOSPolyInterpSOCCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    ipwt::Vector{Matrix{T}}
    is_dual::Bool
end
WSOSPolyInterpSOCCone(R::Int, U::Int, ipwt::Vector{Matrix{T}}) where {T <: Real} = WSOSPolyInterpSOCCone{T}(R, U, ipwt, false)

const MOIOtherConesList(::Type{T}) where {T <: Real} = (
    MOI.NormInfinityCone,
    MOI.NormOneCone,
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    MOI.ExponentialCone,
    MOI.PowerCone{T},
    MOI.GeometricMeanCone,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.LogDetConeTriangle,
    WSOSPolyInterpCone{T},
    WSOSPolyInterpMatCone{T},
    # WSOSPolyInterpSOCCone{T},
    )

const MOIOtherCones{T <: Real} = Union{
    MOI.NormInfinityCone,
    MOI.NormOneCone,
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    MOI.ExponentialCone,
    MOI.PowerCone{T},
    MOI.GeometricMeanCone,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.LogDetConeTriangle,
    WSOSPolyInterpCone{T},
    WSOSPolyInterpMatCone{T},
    WSOSPolyInterpSOCCone{T},
    }

# MOI cones for which no transformation is needed
cone_from_moi(::Type{T}, s::MOI.NormInfinityCone) where {T <: Real} = Cones.EpiNormInf{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.NormOneCone) where {T <: Real} = Cones.EpiNormInf{T}(MOI.dimension(s), true)
cone_from_moi(::Type{T}, s::MOI.SecondOrderCone) where {T <: Real} = Cones.EpiNormEucl{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.RotatedSecondOrderCone) where {T <: Real} = Cones.EpiPerSquare{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.ExponentialCone) where {T <: Real} = Cones.HypoPerLog{T}(3)
cone_from_moi(::Type{T}, s::MOI.GeometricMeanCone) where {T <: Real} = (l = MOI.dimension(s) - 1; Cones.HypoGeomean{T}(fill(inv(l), l)))
cone_from_moi(::Type{T}, s::MOI.PowerCone{T}) where {T <: Real} = Cones.Power{T}([s.exponent, 1 - s.exponent], 1)
cone_from_moi(::Type{T}, s::MOI.PositiveSemidefiniteConeTriangle) where {T <: Real} = Cones.PosSemidefTri{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.LogDetConeTriangle) where {T <: Real} = Cones.HypoPerLogdetTri{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::WSOSPolyInterpCone{T}) where {T <: Real} = Cones.WSOSPolyInterp{T, T}(s.dimension, s.Ps, s.is_dual)
cone_from_moi(::Type{T}, s::WSOSPolyInterpMatCone{T}) where {T <: Real} = Cones.WSOSPolyInterpMat{T}(s.R, s.U, s.ipwt, s.is_dual)
cone_from_moi(::Type{T}, s::WSOSPolyInterpSOCCone{T}) where {T <: Real} = Cones.WSOSPolyInterpSOC{T}(s.R, s.U, s.ipwt, s.is_dual)
cone_from_moi(::Type{T}, s::MOI.AbstractVectorSet) where {T <: Real} = error("MOI set $s is not recognized")
