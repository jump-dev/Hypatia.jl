#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find maximum volume hypercube with edges parallel to the axes inside a polyhedron
defined with l_1, l_infty, or l_2 ball constraints (different to native.jl)
=#

using LinearAlgebra
import JuMP
const MOI = JuMP.MOI
import Hypatia
import Random
using Test

function maxvolume_JuMP(
    ::Type{T},
    n::Int,
    epipernormeucl_constr::Bool, # add an L2 ball constraint, else don't add
    epinorminf_constrs::Bool, # add L1 and Linfty ball constraints, elsle don't add
    ) where {T <: Float64} # TODO support generic reals
    @assert n > 2

    A = randn(n, n)
    # ensure there will be a feasible solution
    x = randn(n)
    gamma = norm(A * x) / sqrt(n)

    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@variable(model, end_pts[1:n])
    JuMP.@objective(model, Max, t)
    JuMP.@constraint(model, vcat(t, end_pts) in MOI.GeometricMeanCone(n + 1))

    if epipernormeucl_constr
        JuMP.@constraint(model, vcat(gamma, A * end_pts) in JuMP.SecondOrderCone())
    end
    if epinorminf_constrs
        JuMP.@constraint(model, vcat(gamma, A * end_pts) in MOI.NormInfinityCone(n + 1))
        JuMP.@constraint(model, vcat(sqrt(n) * gamma, A * end_pts) in MOI.NormOneCone(n + 1))
    end

    return (model, ())
end

function test_maxvolume_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = maxvolume_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

maxvolume_JuMP_fast = [
    (3, true, false),
    (3, false, true),
    (3, true, true),
    (12, true, false),
    (12, false, true),
    (12, true, true),
    (100, true, false),
    (100, false, true),
    (100, true, true),
    (1000, true, false),
    (1000, true, true), # with bridges extended formulation will need to go into slow list
    ]
maxvolume_JuMP_slow = [
    (1000, false, true),
    (2000, true, false),
    (2000, false, true),
    (2000, true, true),
    ]
