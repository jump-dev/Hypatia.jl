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
const DP = DynamicPolynomials
import SumOfSquares
import PolyJuMP
using Test
import Random

const rt2 = sqrt(2)

function muconvexity_JuMP(
    polyfun::Function,
    dom::MU.Domain;
    use_wsos::Bool = true,
    rseed::Int = 1,
    )
    Random.seed!(rseed)

    n = MU.get_dimension(dom)
    DynamicPolynomials.@polyvar x[1:n]
    poly = polyfun(x)
    model = JuMP.Model()
    JuMP.@variable(model, mu)
    JuMP.@objective(model, Max, mu)

    convpoly = poly - 0.5 * mu * sum(x.^2)
    H = DynamicPolynomials.differentiate(convpoly, x, 2)

    if use_wsos
        n = DynamicPolynomials.nvariables(x)
        d = div(maximum(DynamicPolynomials.maxdegree.(H)) + 1, 2)
        (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = true, sample_factor = 200)
        mat_wsos_cone = HYP.WSOSPolyInterpMatCone(n, U, [P0, PWts...])

        JuMP.@constraint(model, [H[i, j](x => pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)

        if dom isa MU.FreeDomain
            JuMP.@constraint(model, H in JuMP.PSDCone())
        else
            JuMP.@constraint(model, H in JuMP.PSDCone(), domain = MU.get_domain_inequalities(dom, x))
        end
    end

    return (model = model, mu = mu)
end

muconvexity1_JuMP() = muconvexity_JuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.FreeDomain(1), use_wsos = true)
muconvexity2_JuMP() = muconvexity_JuMP(x -> sum(x.^4) - sum(x.^2), MU.FreeDomain(3), use_wsos = true)
muconvexity3_JuMP() = muconvexity_JuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.Box([-1.0], [1.0]), use_wsos = true)
muconvexity4_JuMP() = muconvexity_JuMP(x -> sum(x.^4) - sum(x.^2), MU.Ball([5.0, 5.0], 1.0), use_wsos = true) # TODO giving incorrect solution
muconvexity5_JuMP() = muconvexity_JuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.FreeDomain(1), use_wsos = false)
muconvexity6_JuMP() = muconvexity_JuMP(x -> sum(x.^4) - sum(x.^2), MU.FreeDomain(3), use_wsos = false)
muconvexity7_JuMP() = muconvexity_JuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.Box([-1.0], [1.0]), use_wsos = false)
muconvexity8_JuMP() = muconvexity_JuMP(x -> sum(x.^4) - sum(x.^2), MU.Ball([5.0, 5.0], 1.0), use_wsos = false) # TODO giving incorrect solution

function test_muconvexity_JuMP(instance::Tuple{Function,Number}; options)
    (builder, true_mu) = instance
    data = builder()
    JuMP.optimize!(data.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.value(data.mu) ≈ true_mu atol = 1e-4 rtol = 1e-4
end

test_muconvexity_JuMP(; options...) = test_muconvexity_JuMP.([
    (muconvexity1_JuMP, -4),
    (muconvexity2_JuMP, -2),
    (muconvexity3_JuMP, -4),
    (muconvexity4_JuMP, -2),
    (muconvexity5_JuMP, -4),
    (muconvexity6_JuMP, -2),
    (muconvexity7_JuMP, -4),
    (muconvexity8_JuMP, -2),
    ], options = options)

# TODO do we want to keep a random example?
# muconvexity_JuMP() = muconvexity_JuMP(2, sum(rand() * z for z in DynamicPolynomials.monomials(x, 0:4)), MU.FreeDomain(n), true)
# function run_JuMP_muconvexity_rand(; rseed::Int = 1)
#     Random.seed!(rseed)
#
#     n = 2
#     d = 4
#     DynamicPolynomials.@polyvar x[1:n]
#
#     poly = sum(rand() * z for z in DynamicPolynomials.monomials(x, 0:d))
#
#     dom = MU.FreeDomain(n)
#     (term1, prim1, mu1) = run_JuMP_muconvexity(x, poly, dom, true)
#     (term2, prim2, mu2) = run_JuMP_muconvexity(x, poly, dom, false)
#     @test term1 == term2
#     @test prim1 == prim2
#     if term1 == MOI.OPTIMAL || term2 == MOI.OPTIMAL
#         @test mu1 ≈ mu2 atol = 1e-4 rtol = 1e-4
#         mufree = mu1
#     else
#         mufree = -Inf
#     end
#
#     dom = MU.Ball(zeros(n), 1.0)
#     (term1, prim1, mu1) = run_JuMP_muconvexity(x, poly, dom, true)
#     (term2, prim2, mu2) = run_JuMP_muconvexity(x, poly, dom, false)
#     @test term1 == term2 == MOI.OPTIMAL
#     @test prim1 == prim2 == MOI.FEASIBLE_POINT
#     @test mu1 ≈ mu2 atol = 1e-4 rtol = 1e-4
#     muball = mu1
#
#     @test mufree - muball <= 1e-4
# end

# function run_JuMP_muconvexity_a(; use_wsos::Bool = true)
#     DynamicPolynomials.@polyvar x[1:1]
#     poly = (x[1] + 1)^2 * (x[1] - 1)^2
#     dom = MU.FreeDomain(1)
#
#     (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
#     @test term == MOI.OPTIMAL
#     @test prim == MOI.FEASIBLE_POINT
#     @test mu ≈ -4 atol = 1e-4 rtol = 1e-4
# end

# function run_JuMP_muconvexity_b(; use_wsos::Bool = true)
#     n = 3
#     DynamicPolynomials.@polyvar x[1:n]
#     poly = sum(x.^4) - sum(x.^2)
#     dom = MU.FreeDomain(n)
#
#     (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
#     @test term == MOI.OPTIMAL
#     @test prim == MOI.FEASIBLE_POINT
#     @test mu ≈ -2 atol = 1e-4 rtol = 1e-4
# end
#
# function run_JuMP_muconvexity_c(; use_wsos::Bool = true)
#     DynamicPolynomials.@polyvar x[1:1]
#     poly = (x[1] + 1)^2 * (x[1] - 1)^2
#     dom = MU.Box([-1.0], [1.0])
#
#     (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
#     @test term == MOI.OPTIMAL
#     @test prim == MOI.FEASIBLE_POINT
#     @test mu ≈ -4 atol = 1e-4 rtol = 1e-4
# end
#
# function run_JuMP_muconvexity_d(; use_wsos::Bool = true)
#     n = 3 # why was this working? the ball had dimension 2
#     DynamicPolynomials.@polyvar x[1:n]
#     poly = sum(x.^4) - sum(x.^2)
#     dom = MU.Ball([5.0, 5.0], 1.0)
#
#     (term, prim, mu) = run_JuMP_muconvexity(x, poly, dom, use_wsos)
#     @test term == MOI.OPTIMAL
#     @test prim == MOI.FEASIBLE_POINT
#     @test mu ≈ -2 atol = 1e-4 rtol = 1e-4
# end
