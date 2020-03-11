#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find maximum volume hypercube with edges parallel to the axes inside a polyhedron
defined with l_1, l_infty, or l_2 ball constraints (different to native.jl)
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))

struct MaxVolumeJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    n::Int
    epipernormeucl_constr::Bool # add an L2 ball constraint, else don't add
    epinorminf_constrs::Bool # add L1 and Linfty ball constraints, elsle don't add
end

options = ()
example_tests(::Type{MaxVolumeJuMP{Float64}}, ::MinimalInstances) = [
    ((2, true, false), false, options),
    ((2, false, true), false, options),
    ((2, true, true), false, options),
    ]
example_tests(::Type{MaxVolumeJuMP{Float64}}, ::FastInstances) = [
    ((10, true, false), false, options),
    ((10, false, true), false, options),
    ((10, true, true), false, options),
    ((100, true, false), false, options),
    ((100, false, true), false, options),
    ((100, true, true), false, options),
    ((1000, true, false), false, options),
    ((1000, true, true), false, options), # with bridges extended formulation will need to go into slow list
    ]
example_tests(::Type{MaxVolumeJuMP{Float64}}, ::SlowInstances) = [
    ((1000, false, true), false, options),
    ((2000, true, false), false, options),
    ((2000, false, true), false, options),
    ((2000, true, true), false, options),
    ]

function build(inst::MaxVolumeJuMP{T}) where {T <: Float64} # TODO generic reals
    n = inst.n
    A = randn(n, n)
    # ensure there will be a feasible solution
    x = randn(n)
    gamma = norm(A * x) / sqrt(n)

    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@variable(model, end_pts[1:n])
    JuMP.@objective(model, Max, t)
    JuMP.@constraint(model, vcat(t, end_pts) in MOI.GeometricMeanCone(n + 1))

    if inst.epipernormeucl_constr
        JuMP.@constraint(model, vcat(gamma, A * end_pts) in JuMP.SecondOrderCone())
    end
    if inst.epinorminf_constrs
        JuMP.@constraint(model, vcat(gamma, A * end_pts) in MOI.NormInfinityCone(n + 1))
        JuMP.@constraint(model, vcat(sqrt(n) * gamma, A * end_pts) in MOI.NormOneCone(n + 1))
    end

    return model
end

function test_extra(inst::MaxVolumeJuMP, model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

# @testset "MaxVolumeJuMP" for inst in example_tests(MaxVolumeJuMP{Float64}, MinimalInstances()) test(inst...) end

return MaxVolumeJuMP
