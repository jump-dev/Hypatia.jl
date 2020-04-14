#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

check a sufficient condition for pointwise membership of vector valued polynomials in the second order cone
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))

struct SecondOrderPolyJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    polys_name::Symbol
    deg::Int
    is_feas::Bool # whether model should be primal-dual feasible; only for testing
end

secondorderpoly_data = Dict(
    :polys1 => (x -> [x^2 + 2, x]),
    :polys2 => (x -> [2x^2 + 2, x, x]),
    :polys3 => (x -> [x^2 + 2, x, x]),
    :polys4 => (x -> [2 * x^4 + 8 * x^2 + 4, x + 2 + (x + 1)^2, x]),
    :polys5 => (x -> [x, x^2 + x]),
    :polys6 => (x -> [x, x + 1]),
    :polys7 => (x -> [x^2, x]),
    :polys8 => (x -> [x + 2, x]),
    :polys9 => (x -> [x - 1, x, x]),
    )

example_tests(::Type{SecondOrderPolyJuMP{Float64}}, ::MinimalInstances) = [
    ((:polys1, 2, true),),
    ]
example_tests(::Type{SecondOrderPolyJuMP{Float64}}, ::FastInstances) = [
    ((:polys2, 2, true),),
    ((:polys3, 2, true),),
    ((:polys4, 4, true),),
    ((:polys5, 2, false),),
    ((:polys6, 2, false),),
    ((:polys7, 2, false),),
    ((:polys8, 2, false),),
    ((:polys9, 2, false),),
    ]
example_tests(::Type{SecondOrderPolyJuMP{Float64}}, ::SlowInstances) = [
    ]

function build(inst::SecondOrderPolyJuMP{T}) where {T <: Float64} # TODO generic reals
    halfdeg = div(inst.deg + 1, 2)
    (U, pts, Ps) = ModelUtilities.interpolate(ModelUtilities.FreeDomain{Float64}(1), halfdeg)
    vals = secondorderpoly_data[inst.polys_name].(pts)
    l = length(vals[1])
    cone = Hypatia.WSOSInterpEpiNormEuclCone{Float64}(l, U, Ps)

    model = JuMP.Model()
    JuMP.@constraint(model, [v[i] for i in 1:l for v in vals] in cone)

    return model
end

function test_extra(inst::SecondOrderPolyJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == (inst.is_feas ? MOI.OPTIMAL : MOI.INFEASIBLE)
end

return SecondOrderPolyJuMP
