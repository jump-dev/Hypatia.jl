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
include(joinpath(dirname(@__DIR__), "utils", "semialgebraicsets.jl"))

const rt2 = sqrt(2)

function run_JuMP_muconvexity(x::Vector, poly, dom, use_wsos::Bool)
    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true, tolabsopt=1e-8, tolrelopt=1e-8, tolfeas=1e-8))
    @variable(model, mu)
    @objective(model, Max, mu)

    convpoly = poly - 0.5*mu*sum(x.^2)
    H = differentiate(convpoly, x, 2)

    if use_wsos
        n = nvariables(x)
        d = div(maximum(maxdegree.(H)), 2)
        (U, pts, P0, PWts, _) = Hypatia.interpolate(dom, d, sample=true)
        mat_wsos_cone = WSOSPolyInterpMatCone(n, U, [P0, PWts...])

        @constraint(model, [H[i,j](x => pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        if !(dom isa Hypatia.FreeDomain)
            error("implement domains with polyjump syntax")
        end
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        @SDconstraint(model, H >= 0)# TODO domain=get_domain_inequalities(dom, x))
    end

    JuMP.optimize!(model)

    return (JuMP.termination_status(model), JuMP.primal_status(model), JuMP.value(mu))
end

function run_JuMP_muconvexity_rand(;rseed::Int=1)
    n = 2
    d = 4
    @polyvar x[1:n]

    Random.seed!(rseed)
    poly = sum(rand() * z for z in monomials(x, 0:d))
    @show poly

    dom = Hypatia.FreeDomain(n)
    (term1, prim1, mu1) = run_JuMP_muconvexity(x, poly, dom, true)
    (term2, prim2, mu2) = run_JuMP_muconvexity(x, poly, dom, false)
    @test term1 == term2
    @test prim1 == prim2
    if term1 == MOI.OPTIMAL || term2 == MOI.OPTIMAL
        @test mu1 ≈ mu2 atol=1e-4 rtol=1e-4
        mufree = mu1
    else
        mufree = -Inf
    end

    dom = Hypatia.Ball(zeros(n), 1.0)
    (term1, prim1, mu1) = run_JuMP_muconvexity(x, poly, dom, true)
    # (term2, prim2, mu2) = run_JuMP_muconvexity(x, poly, dom, false)
    # @test term1 == term2 == MOI.OPTIMAL
    # @test prim1 == prim2 == MOI.OPTIMAL
    # @test mu1 ≈ mu2 atol=1e-4 rtol=1e-4
    muball = mu1

    @test mufree - muball <= 1e-4
end

function run_JuMP_muconvexity_a(use_wsos::Bool)
    @polyvar x[1:1]
    poly = (x[1] + 1)^2*(x[1] - 1)^2
    dom = Hypatia.FreeDomain(1)

    (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
    @test mu ≈ -4 atol=1e-4 rtol=1e-4
end

function run_JuMP_muconvexity_b(use_wsos::Bool)
    n = 3
    @polyvar x[1:n]
    poly = sum(x.^4) - sum(x.^2)
    dom = Hypatia.FreeDomain(n)

    (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
    @test mu ≈ -2 atol=1e-4 rtol=1e-4
end

function run_JuMP_muconvexity_c(use_wsos::Bool)
    @polyvar x[1:1]
    poly = (x[1] + 1)^2*(x[1] - 1)^2
    dom = Hypatia.Box([-1.0], [1.0])

    (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
    @test mu ≈ -4 atol=1e-4 rtol=1e-4
end

function run_JuMP_muconvexity_d(use_wsos::Bool)
    n = 3
    @polyvar x[1:n]
    poly = sum(x.^4) - sum(x.^2)
    dom = Hypatia.Ball([5.0, 5.0], 1.0)

    (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
    @test mu ≈ -2 atol=1e-4 rtol=1e-4
end
