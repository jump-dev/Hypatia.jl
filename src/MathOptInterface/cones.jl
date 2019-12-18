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
cone_from_moi(::Type{T}, s::MOI.LogDetConeTriangle) where {T <: Real} = Cones.HypoPerLogdetTri{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::MOI.RootDetConeTriangle) where {T <: Real} = Cones.HypoRootdetTri{T, T}(MOI.dimension(s))
cone_from_moi(::Type{T}, s::WSOSInterpNonnegativeCone{T}) where {T <: Real} = Cones.WSOSInterpNonnegative{T, T}(s.U, s.Ps, s.is_dual)
cone_from_moi(::Type{T}, s::WSOSInterpPossemidefTriCone{T}) where {T <: Real} = Cones.WSOSInterpPosSemidefTri{T}(s.R, s.U, s.Ps, s.is_dual)
cone_from_moi(::Type{T}, s::WSOSInterpEpiNormEuclCone{T}) where {T <: Real} = Cones.WSOSInterpEpiNormEucl{T}(s.R, s.U, s.Ps, s.is_dual)
cone_from_moi(::Type{T}, s::MOI.AbstractVectorSet) where {T <: Real} = error("MOI set $s is not recognized")




# TODO delete
const rt2 = sqrt(2)
const rt2i = inv(rt2)
svec_scale(dim) = [(i == j ? 1.0 : rt2) for i in 1:round(Int, sqrt(0.25 + 2 * dim) - 0.5) for j in 1:i]
svec_unscale(dim) = [(i == j ? 1.0 : rt2i) for i in 1:round(Int, sqrt(0.25 + 2 * dim) - 0.5) for j in 1:i]

# PSD cone: convert from smat to svec form (scale off-diagonals)
function build_var_cone(fi::MOI.VectorOfVariables, si::MOI.PositiveSemidefiniteConeTriangle, dim::Int, q::Int)
    IGi = (q + 1):(q + dim)
    VGi = -svec_scale(dim)
    conei = Cones.PosSemidefTri{Float64, Float64}(dim)
    return (IGi, VGi, conei)
end

function build_constr_cone(fi::MOI.VectorAffineFunction{Float64}, si::MOI.PositiveSemidefiniteConeTriangle, dim::Int, q::Int)
    scalevec = svec_scale(dim)
    IGi = [q + vt.output_index for vt in fi.terms]
    VGi = [-vt.scalar_term.coefficient * scalevec[vt.output_index] for vt in fi.terms]
    Ihi = (q + 1):(q + dim)
    Vhi = scalevec .* fi.constants
    conei = Cones.PosSemidefTri{Float64, Float64}(dim)
    return (IGi, VGi, Ihi, Vhi, conei)
end

# logdet cone: convert from smat to svec form (scale off-diagonals)
function build_var_cone(fi::MOI.VectorOfVariables, si::MOI.LogDetConeTriangle, dim::Int, q::Int)
    IGi = (q + 1):(q + dim)
    VGi = vcat(-1.0, -1.0, -svec_scale(dim - 2))
    conei = Cones.HypoPerLogdetTri{Float64, Float64}(dim)
    return (IGi, VGi, conei)
end

function build_constr_cone(fi::MOI.VectorAffineFunction{Float64}, si::MOI.LogDetConeTriangle, dim::Int, q::Int)
    scalevec = vcat(1.0, 1.0, svec_scale(dim - 2))
    IGi = [q + vt.output_index for vt in fi.terms]
    VGi = [-vt.scalar_term.coefficient * scalevec[vt.output_index] for vt in fi.terms]
    Ihi = (q + 1):(q + dim)
    Vhi = scalevec .* fi.constants
    conei = Cones.HypoPerLogdetTri{Float64, Float64}(dim)
    return (IGi, VGi, Ihi, Vhi, conei)
end





untransform_cone_vec(cone::Cones.Cone, ::AbstractVector, z::AbstractVector) = nothing

function untransform_cone_vec(cone::Cones.PosSemidefTri, s::AbstractVector, z::AbstractVector)
    ModelUtilities.svec_to_vec!(s)
    ModelUtilities.svec_to_vec!(z)
    return nothing
end

function untransform_cone_vec(cone::Cones.HypoPerLogdetTri, s::AbstractVector, z::AbstractVector)
    @views ModelUtilities.svec_to_vec!(s[3:end])
    @views ModelUtilities.svec_to_vec!(z[3:end])
    return nothing
end

function untransform_cone_vec(cone::Cones.HypoRootdetTri, s::AbstractVector, z::AbstractVector)
    @views ModelUtilities.svec_to_vec!(s[2:end])
    @views ModelUtilities.svec_to_vec!(z[2:end])
    return nothing
end

# TODO scale for WSOSInterpPossemidefTriCone?
