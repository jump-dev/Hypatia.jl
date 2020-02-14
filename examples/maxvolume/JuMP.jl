#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find maximum volume hypercube with edges parallel to the axes inside a polyhedron
defined with l_1, l_infty, or l_2 constraints (different to native.jl)
=#

using LinearAlgebra
import JuMP
const MOI = JuMP.MOI
import Hypatia
import Random
using Test

function maxvolumeJuMP(
    n::Int;
    epipernormeucl_constr::Bool = false,
    epinorminf_constr::Bool = false,
    epinorminfdual_constr::Bool = false,
    )
    @assert n > 2
    A = randn(n, n)
    # ensure there will be a feasible solution
    x = randn(n)
    gamma = norm(A * x) / sqrt(n)

    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@variable(model, end_pts[1:n])
    JuMP.@objective(model, Max, t)
    if epipernormeucl_constr
        JuMP.@constraint(model, vcat(gamma, A * end_pts) in JuMP.SecondOrderCone())
    end
    if epinorminf_constr
        JuMP.@constraint(model, vcat(gamma, A * end_pts) in MOI.NormInfinityCone(n + 1))
    end
    if epinorminfdual_constr
        JuMP.@constraint(model, vcat(sqrt(n) * gamma, A * end_pts) in MOI.NormOneCone(n + 1))
    end
    JuMP.@constraint(model, vcat(t, end_pts) in MOI.GeometricMeanCone(n + 1))

    return (model = model,)
end

maxvolumeJuMP1() = maxvolumeJuMP(3, epipernormeucl_constr = true)
maxvolumeJuMP2() = maxvolumeJuMP(3, epipernormeucl_constr = true, epinorminf_constr = true, epinorminfdual_constr = true)
maxvolumeJuMP3() = maxvolumeJuMP(6, epipernormeucl_constr = true)
maxvolumeJuMP4() = maxvolumeJuMP(6, epipernormeucl_constr = true, epinorminf_constr = true, epinorminfdual_constr = true)
maxvolumeJuMP5() = maxvolumeJuMP(25, epipernormeucl_constr = true)
maxvolumeJuMP6() = maxvolumeJuMP(25, epipernormeucl_constr = true, epinorminf_constr = true, epinorminfdual_constr = true)

function test_maxvolumeJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_maxvolumeJuMP_all(; options...) = test_maxvolumeJuMP.([
    maxvolumeJuMP1,
    maxvolumeJuMP2,
    maxvolumeJuMP3,
    maxvolumeJuMP4,
    maxvolumeJuMP5,
    maxvolumeJuMP6,
    ], options = options)

test_maxvolumeJuMP(; options...) = test_maxvolumeJuMP.([
    maxvolumeJuMP1,
    maxvolumeJuMP2,
    ], options = options)
