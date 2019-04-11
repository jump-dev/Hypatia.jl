#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

find parameter of convexity mu for a given polynomial p(x)
ie the largest mu such that p(x) - mu/2*||x||^2 is convex everywhere on given domain
see https://en.wikipedia.org/wiki/Convex_function#Strongly_convex_functions
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MU = HYP.ModelUtilities

import MathOptInterface
const MOI = MathOptInterface
import JuMP
import MultivariatePolynomials
import DynamicPolynomials
import SumOfSquares
import PolyJuMP
using Test
import Random
using TimerOutputs
using LinearAlgebra

const rt2 = sqrt(2)

# function bilinear_P(P, n) # TODO delete, just a kronecker product
#     (U, L) = size(P)
#     naive_P = zeros(div(n * (n + 1), 2) * U, n * L)
#     @show U, L, size(naive_P)
#     col = 1
#     for i in 1:n
#         row = 1
#         for j in 1:n, k in j:n
#             if i in (j, k)
#                 naive_P[row:(row + U - 1), col:(col + L - 1)] = P
#             end
#             row += U
#         end
#         col += L
#     end
#     return naive_P
# end

function run_JuMP_muconvexity(x::Vector, poly, dom, use_wsos::Bool)
    Random.seed!(1)
    reset_timer!(Hypatia.to)

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@variable(model, mu)
    JuMP.@objective(model, Max, mu)

    convpoly = poly - 0.5 * mu * sum(x.^2)
    H = DynamicPolynomials.differentiate(convpoly, x, 2)

    if use_wsos
        n = DynamicPolynomials.nvariables(x)
        d = div(maximum(DynamicPolynomials.maxdegree.(H)) + 1, 2)
        (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = true)
        @show cond(P0)
        mat_wsos_cone = HYP.WSOSPolyInterpMatCone(n, U, [P0, PWts...])

        JuMP.@constraint(model, [H[i, j](x => pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)

        if dom isa MU.FreeDomain
            JuMP.@constraint(model, H in SumOfSquares.PSDCone())
        else
            JuMP.@constraint(model, H in SumOfSquares.PSDCone(), domain = MU.get_domain_inequalities(dom, x))
        end
    end

    JuMP.optimize!(model)

    return (JuMP.termination_status(model), JuMP.primal_status(model), JuMP.value(mu))
end

function randconvfun(n, d)
    L = binomial(n + d, n)
    randmat = randn(L, L)
    randmat = Symmetric(randmat * randmat')
    DynamicPolynomials.@polyvar x[1:n]
    monos = DynamicPolynomials.monomials(x, 0:d)
    poly = monos' * randmat * monos
    return poly
end

function run_JuMP_muconvexity_scalar(x::Vector, poly, dom, use_wsos::Bool)
    Random.seed!(1)
    reset_timer!(Hypatia.to)

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@variable(model, mu)
    JuMP.@objective(model, Max, mu)

    convpoly = poly - 0.5 * mu * sum(x.^2)
    H = DynamicPolynomials.differentiate(convpoly, x, 2)

    n = DynamicPolynomials.nvariables(x)
    DynamicPolynomials.@polyvar y[1:n]

    conv_condition = y' * H * y

    fulldim = false

    if use_wsos
        d = div(maximum(DynamicPolynomials.maxdegree.(H)) + 1, 2)
        (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = true)
        if fulldim
            naive_domain = MU.FreeDomain(2n)
            (naive_U, naive_pts, naive_P0, naive_PWts) = MU.interpolate(naive_domain, d + 1, sample = true)
            @show cond(naive_P0)
        else
            (naive_U, naive_pts, naive_P0, naive_PWts) = MU.bilinear_terms(U, pts, P0, PWts, n)
        end
        @show cond(naive_P0)
        wsos_cone = HYP.WSOSPolyInterpCone(naive_U, [naive_P0, naive_PWts...])

        @show DynamicPolynomials.variables(conv_condition) # order OK
        @show size(naive_pts)
        @show conv_condition(naive_pts[1, :])

        JuMP.@constraint(model, [conv_condition(naive_pts[u, :]) for u in 1:naive_U] in wsos_cone)

    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)

        if dom isa MU.FreeDomain
            JuMP.@constraint(model, H in SumOfSquares.PSDCone())
        else
            JuMP.@constraint(model, H in SumOfSquares.PSDCone(), domain = MU.get_domain_inequalities(dom, x))
        end
    end

    JuMP.optimize!(model)

    return (JuMP.termination_status(model), JuMP.primal_status(model), JuMP.value(mu))
end


function run_JuMP_muconvexity_rand(; rseed::Int = 1)
    Random.seed!(rseed)

    n = 2
    d = 6
    # DynamicPolynomials.@polyvar x[1:n]

    # poly = sum(rand() * z for z in DynamicPolynomials.monomials(x, 0:d))
    poly = randconvfun(n, d)
    x = DynamicPolynomials.variables(poly)

    dom = MU.FreeDomain(n)
    # (term1, prim1, mu1) = run_JuMP_muconvexity(x, poly, dom, true)
    (term2, prim2, mu2) = run_JuMP_muconvexity_scalar(x, poly, dom, true)
    # # (term2, prim2, mu2) = run_JuMP_muconvexity(x, poly, dom, false)
    @test term1 == term2
    @test prim1 == prim2
    if term1 == MOI.OPTIMAL || term2 == MOI.OPTIMAL
        @test mu1 ≈ mu2 atol = 1e-4 rtol = 1e-4
        mufree = mu1
    else
        mufree = -Inf
    end

    # dom = MU.Ball(zeros(n), 1.0)
    # (term1, prim1, mu1) = run_JuMP_muconvexity(x, poly, dom, true)
    # (term2, prim2, mu2) = run_JuMP_muconvexity(x, poly, dom, false)
    # @test term1 == term2 == MOI.OPTIMAL
    # @test prim1 == prim2 == MOI.FEASIBLE_POINT
    # @test mu1 ≈ mu2 atol = 1e-4 rtol = 1e-4
    # muball = mu1

    # @test mufree - muball <= 1e-4
end

function run_JuMP_muconvexity_a(; use_wsos::Bool = true)
    DynamicPolynomials.@polyvar x[1:1]
    poly = (x[1] + 1)^2 * (x[1] - 1)^2
    dom = MU.FreeDomain(1)

    (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
    # (term, prim, mu) = run_JuMP_muconvexity_scalar(x, poly, dom, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
    @test mu ≈ -4 atol = 1e-4 rtol = 1e-4
end

function run_JuMP_muconvexity_b(; use_wsos::Bool = true)
    n = 3
    DynamicPolynomials.@polyvar x[1:n]
    poly = sum(x.^4) - sum(x.^2)
    dom = MU.FreeDomain(n)

    (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
    # (term, prim, mu) = run_JuMP_muconvexity_scalar(x, poly, dom, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
    @test mu ≈ -2 atol = 1e-4 rtol = 1e-4
end

function run_JuMP_muconvexity_c(; use_wsos::Bool = true)
    DynamicPolynomials.@polyvar x[1:1]
    poly = (x[1] + 1)^2 * (x[1] - 1)^2
    dom = MU.Box([-1.0], [1.0])

    # (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
    (term, prim, mu) = run_JuMP_muconvexity_scalar(x, poly, dom, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
    @test mu ≈ -4 atol = 1e-4 rtol = 1e-4
end

function run_JuMP_muconvexity_d(; use_wsos::Bool = true)
    n = 3
    DynamicPolynomials.@polyvar x[1:n]
    poly = sum(x.^4) - sum(x.^2)
    dom = MU.Ball([5.0, 5.0], 1.0)

    # (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
    (term, prim, mu) = run_JuMP_muconvexity_scalar(x, poly, dom, use_wsos)
    @test term == MOI.OPTIMAL
    @test prim == MOI.FEASIBLE_POINT
    @test mu ≈ -2 atol = 1e-4 rtol = 1e-4
end
