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

export WSOSInterpPossemidefTriCone

struct WSOSInterpPossemidefTriCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    is_dual::Bool
end
WSOSInterpPossemidefTriCone(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpPossemidefTriCone{T}(R, U, Ps, false)

export WSOSInterpEpiNormEuclCone

struct WSOSInterpEpiNormEuclCone{T <: Real} <: MOI.AbstractVectorSet
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    is_dual::Bool
end
WSOSInterpEpiNormEuclCone(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpEpiNormEuclCone{T}(R, U, Ps, false)

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
    MOI.PowerCone{T},
    MOI.GeometricMeanCone,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.LogDetConeTriangle,
    WSOSInterpNonnegativeCone{T},
    WSOSInterpPossemidefTriCone{T},
    WSOSInterpEpiNormEuclCone{T},
    }

# MOI cones for which no transformation is needed
cone_from_moi(::Type{T}, s::MOI.NormInfinityCone) where {T <: Real} = Cones.EpiNormInf{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.NormOneCone) where {T <: Real} = Cones.EpiNormInf{T, T}(MOI.dimension(s), true)
cone_from_moi(::Type{T}, s::MOI.SecondOrderCone) where {T <: Real} = Cones.EpiNormEucl{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.RotatedSecondOrderCone) where {T <: Real} = Cones.EpiPerSquare{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.ExponentialCone) where {T <: Real} = Cones.HypoPerLog{T}(3)
cone_from_moi(::Type{T}, s::MOI.GeometricMeanCone) where {T <: Real} = (l = MOI.dimension(s) - 1; Cones.HypoGeomean{T}(fill(inv(l), l)))
cone_from_moi(::Type{T}, s::MOI.PowerCone{T}) where {T <: Real} = Cones.Power{T}([s.exponent, 1 - s.exponent], 1)
cone_from_moi(::Type{T}, s::MOI.PositiveSemidefiniteConeTriangle) where {T <: Real} = Cones.PosSemidefTri{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.LogDetConeTriangle) where {T <: Real} = Cones.HypoPerLogdetTri{T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::WSOSInterpNonnegativeCone{T}) where {T <: Real} = Cones.WSOSInterpNonnegative{T, T}(s.U, s.Ps, s.is_dual)
cone_from_moi(::Type{T}, s::WSOSInterpPossemidefTriCone{T}) where {T <: Real} = Cones.WSOSInterpPosSemidefTri{T}(s.R, s.U, s.Ps, s.is_dual)
cone_from_moi(::Type{T}, s::WSOSInterpEpiNormEuclCone{T}) where {T <: Real} = Cones.WSOSInterpEpiNormEucl{T}(s.R, s.U, s.Ps, s.is_dual)
cone_from_moi(::Type{T}, s::MOI.AbstractVectorSet) where {T <: Real} = error("MOI set $s is not recognized")
