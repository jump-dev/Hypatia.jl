#=
Copyright 2018, Chris Coey and contributors

definitions of conic sets not already defined by MathOptInterface
and functions for converting between Hypatia and MOI cone definitions
=#

export WSOSPolyInterpCone

struct WSOSPolyInterpCone <: MOI.AbstractVectorSet
    dimension::Int
    ipwt::Vector{Matrix{Float64}}
    isdual::Bool
end
WSOSPolyInterpCone(dimension::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpCone(dimension, ipwt, false)


export WSOSPolyInterpMatCone

struct WSOSPolyInterpMatCone <: MOI.AbstractVectorSet
    r::Int
    u::Int
    ipwt::Vector{Matrix{Float64}}
    isdual::Bool
end
WSOSPolyInterpMatCone(r::Int, u::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpMatCone(r, u, ipwt, false)


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
)

# MOI cones for which no transformation is needed
conefrommoi(s::MOI.SecondOrderCone) = Cones.EpiNormEucl(MOI.dimension(s))
conefrommoi(s::MOI.RotatedSecondOrderCone) = Cones.EpiPerSquare(MOI.dimension(s))
conefrommoi(s::MOI.ExponentialCone) = Cones.HypoPerLog()
conefrommoi(s::MOI.GeometricMeanCone) = (l = MOI.dimension(s) - 1; Cones.HypoGeomean(fill(1.0/l, l)))
conefrommoi(s::MOI.PowerCone{Float64}) = Cones.EpiPerPower(inv(s.exponent))
conefrommoi(s::WSOSPolyInterpCone) = Cones.WSOSPolyInterp(s.dimension, s.ipwt, s.isdual)
conefrommoi(s::WSOSPolyInterpMatCone) = Cones.WSOSPolyInterpMat(s.r, s.u, s.ipwt, s.isdual)
conefrommoi(s::MOI.AbstractVectorSet) = error("MOI set $s is not recognized")

function buildvarcone(fi::MOI.VectorOfVariables, si::MOI.AbstractVectorSet, dim::Int, q::Int)
    IGi = q+1:q+dim
    VGi = -ones(dim)
    prmtvi = conefrommoi(si)
    return (IGi, VGi, prmtvi)
end

function buildconstrcone(fi::MOI.VectorAffineFunction{Float64}, si::MOI.AbstractVectorSet, dim::Int, q::Int)
    IGi = [q + vt.output_index for vt in fi.terms]
    VGi = [-vt.scalar_term.coefficient for vt in fi.terms]
    Ihi = q+1:q+dim
    Vhi = fi.constants
    prmtvi = conefrommoi(si)
    return (IGi, VGi, Ihi, Vhi, prmtvi)
end

# MOI cones requiring transformations (eg rescaling, changing order)
# TODO later remove if MOI gets scaled triangle sets
const rt2 = sqrt(2)
const rt2i = inv(rt2)

svecscale(dim) = [(i == j ? 1.0 : rt2) for i in 1:round(Int, sqrt(0.25 + 2*dim) - 0.5) for j in 1:i]
svecunscale(dim) = [(i == j ? 1.0 : rt2i) for i in 1:round(Int, sqrt(0.25 + 2*dim) - 0.5) for j in 1:i]

# PSD cone: convert from smat to svec form (scale off-diagonals)
function buildvarcone(fi::MOI.VectorOfVariables, si::MOI.PositiveSemidefiniteConeTriangle, dim::Int, q::Int)
    IGi = q+1:q+dim
    VGi = -svecscale(dim)
    prmtvi = Cones.PosSemidef(dim)
    return (IGi, VGi, prmtvi)
end

function buildconstrcone(fi::MOI.VectorAffineFunction{Float64}, si::MOI.PositiveSemidefiniteConeTriangle, dim::Int, q::Int)
    scalevec = svecscale(dim)
    IGi = [q + vt.output_index for vt in fi.terms]
    VGi = [-vt.scalar_term.coefficient*scalevec[vt.output_index] for vt in fi.terms]
    Ihi = q+1:q+dim
    Vhi = scalevec .* fi.constants
    prmtvi = Cones.PosSemidef(dim)
    return (IGi, VGi, Ihi, Vhi, prmtvi)
end

# logdet cone: convert from smat to svec form (scale off-diagonals)
function buildvarcone(fi::MOI.VectorOfVariables, si::MOI.LogDetConeTriangle, dim::Int, q::Int)
    IGi = q+1:q+dim
    VGi = vcat(-1.0, -1.0, -svecscale(dim-2))
    prmtvi = Cones.HypoPerLogdet(dim)
    return (IGi, VGi, prmtvi)
end

function buildconstrcone(fi::MOI.VectorAffineFunction{Float64}, si::MOI.LogDetConeTriangle, dim::Int, q::Int)
    scalevec = vcat(1.0, 1.0, svecscale(dim-2))
    IGi = [q + vt.output_index for vt in fi.terms]
    VGi = [-vt.scalar_term.coefficient*scalevec[vt.output_index] for vt in fi.terms]
    Ihi = q+1:q+dim
    Vhi = scalevec .* fi.constants
    prmtvi = Cones.HypoPerLogdet(dim)
    return (IGi, VGi, Ihi, Vhi, prmtvi)
end
