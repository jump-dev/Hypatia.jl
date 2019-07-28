#=
Copyright 2018, Chris Coey and contributors

definitions of conic sets not already defined by MathOptInterface
and functions for converting between Hypatia and MOI cone definitions
=#

export WSOSPolyInterpCone

struct WSOSPolyInterpCone <: MOI.AbstractVectorSet # real case only
    dimension::Int
    Ps::Vector{Matrix{Float64}}
    is_dual::Bool
end
WSOSPolyInterpCone(dimension::Int, Ps::Vector{Matrix{Float64}}) = WSOSPolyInterpCone(dimension, Ps, false)

export WSOSPolyInterpMatCone # TODO rename

struct WSOSPolyInterpMatCone <: MOI.AbstractVectorSet
    R::Int
    U::Int
    ipwt::Vector{Matrix{Float64}}
    is_dual::Bool
end
WSOSPolyInterpMatCone(R::Int, U::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpMatCone(R, U, ipwt, false)

export WSOSPolyInterpSOCCone # TODO rename

struct WSOSPolyInterpSOCCone <: MOI.AbstractVectorSet
    R::Int
    U::Int
    ipwt::Vector{Matrix{Float64}}
    is_dual::Bool
end
WSOSPolyInterpSOCCone(R::Int, U::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpSOCCone(R, U, ipwt, false)

MOIOtherCones = (
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    MOI.ExponentialCone,
    MOI.PowerCone{Float64},
    MOI.GeometricMeanCone,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.LogDetConeTriangle,
    WSOSPolyInterpCone,
    WSOSPolyInterpMatCone,
    WSOSPolyInterpSOCCone,
)

# MOI cones for which no transformation is needed
cone_from_moi(s::MOI.SecondOrderCone) = Cones.EpiNormEucl{Float64}(MOI.dimension(s))
cone_from_moi(s::MOI.RotatedSecondOrderCone) = Cones.EpiPerSquare{Float64}(MOI.dimension(s))
cone_from_moi(s::MOI.ExponentialCone) = Cones.HypoPerLog{Float64}(3)
cone_from_moi(s::MOI.GeometricMeanCone) = (l = MOI.dimension(s) - 1; Cones.HypoGeomean{Float64}(fill(inv(l), l)))
cone_from_moi(s::MOI.PowerCone{Float64}) = Cones.EpiPerPower{Float64}(inv(s.exponent))
cone_from_moi(s::WSOSPolyInterpCone) = Cones.WSOSPolyInterp{Float64, Float64}(s.dimension, s.Ps, s.is_dual)
cone_from_moi(s::WSOSPolyInterpMatCone) = Cones.WSOSPolyInterpMat{Float64}(s.R, s.U, s.ipwt, s.is_dual)
cone_from_moi(s::WSOSPolyInterpSOCCone) = Cones.WSOSPolyInterpSOC{Float64}(s.R, s.U, s.ipwt, s.is_dual)
cone_from_moi(s::MOI.AbstractVectorSet) = error("MOI set $s is not recognized")

function build_var_cone(fi::MOI.VectorOfVariables, si::MOI.AbstractVectorSet, dim::Int, q::Int)
    IGi = (q + 1):(q + dim)
    VGi = -ones(dim)
    conei = cone_from_moi(si)
    return (IGi, VGi, conei)
end

function build_constr_cone(fi::MOI.VectorAffineFunction{Float64}, si::MOI.AbstractVectorSet, dim::Int, q::Int)
    IGi = [q + vt.output_index for vt in fi.terms]
    VGi = [-vt.scalar_term.coefficient for vt in fi.terms]
    Ihi = (q + 1):(q + dim)
    Vhi = fi.constants
    conei = cone_from_moi(si)
    return (IGi, VGi, Ihi, Vhi, conei)
end
