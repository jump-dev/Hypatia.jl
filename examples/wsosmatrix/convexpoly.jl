#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

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

function run_JuMP_convexpoly(x::Vector, poly, dom, use_wsos::Bool)
    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))

    H = differentiate(poly, x, 2)

    @assert dom isa Hypatia.FreeDomain

    if use_wsos
        n = nvariables(x)
        d = div(maximum(maxdegree.(H)), 2)
        (U, pts, P0, PWts, _) = Hypatia.interpolate(dom, d, sample=false)

        mat_wsos_cone = WSOSPolyInterpMatCone(n, U, [P0, PWts...])
        @constraint(model, [AffExpr(H[i,j](x => pts[u, :]) * (i == j ? 1.0 : rt2)) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)

        L = size(P0, 2)

        t = x[1]
        Z = [1 + 0t, t]
        for l in 3:L
            push!(Z, 2*t*Z[l-1] - Z[l-2])
        end
        @show Z

        P0big = [differentiate(Z[l], t, 2)(x => pts[u, :]) for u in 1:U, l in 3:L]
        @show P0big

        sqrU = abs2.(pts)
        Hypatia.getintdir_prmtv!(arr::AbstractVector{Float64}, prmtv::WSOSConvexPolyInterp) = (@. arr = sqrU; arr)

        conv_wsos_cone = WSOSConvexPolyInterpCone(n, U, [P0big])
        @constraint(model, [AffExpr(poly(x => pts[u, :])) for u in 1:U] in conv_wsos_cone)
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)

        if dom isa Hypatia.FreeDomain
            @constraint(model, H in PSDCone())
        else
            @constraint(model, H in PSDCone(), domain=get_domain_inequalities(dom, x))
        end
    end

    JuMP.optimize!(model)

    return (JuMP.termination_status(model), JuMP.primal_status(model))
end

function run_JuMP_convexpoly_rand(; rseed::Int=1)
    n = 1
    d = 8
    @polyvar x[1:n]

    Random.seed!(rseed)
    poly = sum(randn() * z for z in monomials(x, 0:d))

    dom = Hypatia.FreeDomain(n)
    (term1, prim1) = run_JuMP_convexpoly(x, poly, dom, true)
    # (term2, prim2) = run_JuMP_convexpoly(x, poly, dom, false)
    # @test term1 == term2
    # @test prim1 == prim2

    # dom = Hypatia.Ball(zeros(n), 1.0)
    # (term1, prim1) = run_JuMP_convexpoly(x, poly, dom, true)
    # (term2, prim2) = run_JuMP_convexpoly(x, poly, dom, false)
    # @test term1 == term2 == MOI.OPTIMAL
    # @test prim1 == prim2 == MOI.FEASIBLE_POINT
end

function run_JuMP_convexpoly_a(; use_wsos::Bool=true)
    @polyvar x[1:1]
    poly = (x[1] + 1)^2*(x[1] - 1)^2
    dom = Hypatia.FreeDomain(1)

    (term, prim) = run_JuMP_convexpoly(x, poly, dom, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
end

function run_JuMP_convexpoly_b(; use_wsos::Bool=true)
    n = 3
    @polyvar x[1:n]
    poly = sum(x.^4) - sum(x.^2)
    dom = Hypatia.FreeDomain(n)

    (term, prim) = run_JuMP_convexpoly(x, poly, dom, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
end
