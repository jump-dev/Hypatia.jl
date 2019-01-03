#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

find parameter of convexity mu for a given polynomial p(x)
ie the largest mu such that p(x) - mu/2*||x||^2 is convex everywhere on given domain
see https://en.wikipedia.org/wiki/Convex_function#Strongly_convex_functions
=#

using JuMP
using MathOptInterface
MOI = MathOptInterface
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using SumOfSquares
using PolyJuMP
using Test
using Random

const rt2 = sqrt(2)

function run_JuMP_muconvexity(x::Vector, poly, use_wsos::Bool)
    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true, tolabsopt=1e-8, tolrelopt=1e-8, tolfeas=1e-8))
    @variable(model, mu)
    @objective(model, Max, mu)

    convpoly = poly - 0.5*mu*sum(x.^2)
    H = differentiate(convpoly, x, 2)

    if use_wsos
        n = nvariables(x)
        d = div(maximum(maxdegree.(H)), 2)
        dom = Hypatia.FreeDomain(n)
        (U, pts, P0, _, _) = Hypatia.interpolate(dom, d, sample=false)
        mat_wsos_cone = WSOSPolyInterpMatCone(n, U, [P0])

        @constraint(model, [H[i,j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        @constraint(model, H in PSDCone())
    end

    JuMP.optimize!(model)

    return (JuMP.termination_status(model), JuMP.primal_status(model), JuMP.value(mu))
end

function run_JuMP_muconvexity_a(use_wsos::Bool)
    @polyvar x[1:1]
    poly = (x[1] + 1)^2*(x[1] - 1)^2

    (term, prim, mu) = run_JuMP_muconvexity(x, poly, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
    @test mu ≈ -4 atol=1e-4 rtol=1e-4
end

function run_JuMP_muconvexity_b(use_wsos::Bool)
    n = 3
    @polyvar x[1:n]
    poly = sum(x.^4) - sum(x.^2)

    (term, prim, mu) = run_JuMP_muconvexity(x, poly, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
    @test mu ≈ -2 atol=1e-4 rtol=1e-4
end

function run_JuMP_muconvexity_rand(rseed::Int=1)
    n = 2
    d = 4
    @polyvar x[1:n]
    Random.seed!(rseed)
    poly = sum(rand() * z for z in monomials(x, 0:d))
    @show poly

    (term1, prim1, mu1) = run_JuMP_muconvexity(x, poly, true)
    (term2, prim2, mu2) = run_JuMP_muconvexity(x, poly, false)
    @test term1 == term2
    @test prim1 == prim2
    if term1 == MOI.OPTIMAL || term2 == MOI.OPTIMAL
        @test mu1 ≈ mu2 atol=1e-4 rtol=1e-4
    end
end
