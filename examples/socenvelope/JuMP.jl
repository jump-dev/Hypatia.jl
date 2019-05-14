#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find a polynomial f such that f >= \sum_i g_i^2 where g_i are arbitrary polynomials, and the volumne under f is minimized

# TODO add scalar formulation (model utilities code for sparse basis not in master)

=#

using LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

function socenvelopeJuMP(n::Int, deg::Int, npolys::Int)
    vec_length = npolys + 1
    dom = MU.FreeDomain(n)
    DP.@polyvar x[1:n]
    halfdeg = div(deg + 1, 2)
    (U, pts, P0, _, w) = MU.interpolate(dom, halfdeg, sample = false, calc_w = true)
    lagrange_polys = MU.recover_lagrange_polys(pts, 2 * halfdeg)

    model = JuMP.Model()
    JuMP.@variable(model, f[1:U])
    JuMP.@objective(model, Min, dot(w, f))

    L = size(P0, 2)
    polys = P0[:, 1:L] * rand(-9:9, L, npolys)

    fpoly = dot(f, lagrange_polys)
    rand_polys = [dot(polys[:, i], lagrange_polys) for i in 1:npolys]
    cone = HYP.WSOSPolyInterpSOCCone(vec_length, U, [P0])
    JuMP.@constraint(model, vcat(f, [polys[:, i] for i in 1:npolys]...) in cone)

    return (model = model,)
end

socenvelopeJuMP1() = socenvelopeJuMP(2, 2, 2)
socenvelopeJuMP2() = socenvelopeJuMP(2, 1, 3)

function test_socenvelopeJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_socenvelopeJuMP(; options...) = test_socenvelopeJuMP.([
    socenvelopeJuMP1,
    socenvelopeJuMP2,
    ], options = options)
