#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find parameter of convexity mu for a given polynomial p(x)
ie the largest mu such that p(x) - mu/2*||x||^2 is convex everywhere on given domain
see https://en.wikipedia.org/wiki/Convex_function#Strongly_convex_functions
=#

using Test
import Random
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
const DP = DynamicPolynomials
import SumOfSquares
import PolyJuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

const rt2 = sqrt(2)

function muconvexityJuMP(
    polyfun::Function,
    dom::MU.Domain;
    use_wsos::Bool = true,
    )
    n = MU.get_dimension(dom)
    DP.@polyvar x[1:n]
    poly = polyfun(x)

    model = JuMP.Model()
    JuMP.@variable(model, mu)
    JuMP.@objective(model, Max, mu)

    convpoly = poly - 0.5 * mu * sum(x.^2)
    H = DP.differentiate(convpoly, x, 2)

    if use_wsos
        n = DP.nvariables(x)
        d = div(maximum(DP.maxdegree.(H)) + 1, 2)
        (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = true, sample_factor = 100)
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

muconvexityJuMP1() = muconvexityJuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.FreeDomain(1), use_wsos = true)
muconvexityJuMP2() = muconvexityJuMP(x -> sum(x.^4) - sum(x.^2), MU.FreeDomain(3), use_wsos = true)
muconvexityJuMP3() = muconvexityJuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.Box([-1.0], [1.0]), use_wsos = true)
muconvexityJuMP4() = muconvexityJuMP(x -> sum(x.^4) - sum(x.^2), MU.Ball([5.0, 5.0], 1.0), use_wsos = true) # TODO giving incorrect solution
muconvexityJuMP5() = muconvexityJuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.FreeDomain(1), use_wsos = false)
muconvexityJuMP6() = muconvexityJuMP(x -> sum(x.^4) - sum(x.^2), MU.FreeDomain(3), use_wsos = false)
muconvexityJuMP7() = muconvexityJuMP(x -> (x[1] + 1)^2 * (x[1] - 1)^2, MU.Box([-1.0], [1.0]), use_wsos = false)
muconvexityJuMP8() = muconvexityJuMP(x -> sum(x.^4) - sum(x.^2), MU.Ball([5.0, 5.0], 1.0), use_wsos = false) # TODO giving incorrect solution
# function muconvexityJuMP9()
#     Random.seed!(1234)
#     n = 2
#     d = 4
#     DP.@polyvar x[1:n]
#     poly = sum(rand() * z for z in DP.monomials(x, 0:d))
#     dom = MU.FreeDomain(n)
#     return muconvexityJuMP(x, x -> poly(x), dom, use_wsos = true)
# end

function test_muconvexityJuMP(instance::Tuple{Function,Number}; options, rseed::Int = 1)
    Random.seed!(rseed)
    (instance, true_mu) = instance
    d = instance()
    JuMP.optimize!(d.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.value(d.mu) ≈ true_mu atol = 1e-4 rtol = 1e-4
end

test_muconvexityJuMP(; options...) = test_muconvexityJuMP.([
    (muconvexityJuMP1, -4),
    (muconvexityJuMP2, -2),
    (muconvexityJuMP3, -4),
    (muconvexityJuMP4, -2),
    (muconvexityJuMP5, -4),
    (muconvexityJuMP6, -2),
    (muconvexityJuMP7, -4),
    (muconvexityJuMP8, -2),
    ], options = options)

# TODO do we want to keep a random example?
# muconvexityJuMP() = muconvexityJuMP(2, sum(rand() * z for z in DP.monomials(x, 0:4)), MU.FreeDomain(n), true)
# function run_muconvexityJuMP_rand(; rseed::Int = 1)
#     Random.seed!(rseed)
#
#     n = 2
#     d = 4
#     DP.@polyvar x[1:n]
#
#     poly = sum(rand() * z for z in DP.monomials(x, 0:d))
#
#     dom = MU.FreeDomain(n)
#     (term1, prim1, mu1) = run_muconvexityJuMP(x, poly, dom, true)
#     (term2, prim2, mu2) = run_muconvexityJuMP(x, poly, dom, false)
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
#     (term1, prim1, mu1) = run_muconvexityJuMP(x, poly, dom, true)
#     (term2, prim2, mu2) = run_muconvexityJuMP(x, poly, dom, false)
#     @test term1 == term2 == MOI.OPTIMAL
#     @test prim1 == prim2 == MOI.FEASIBLE_POINT
#     @test mu1 ≈ mu2 atol = 1e-4 rtol = 1e-4
#     muball = mu1
#
#     @test mufree - muball <= 1e-4
# end

# function run_muconvexityJuMP_a(; use_wsos::Bool = true)
#     DP.@polyvar x[1:1]
#     poly = (x[1] + 1)^2 * (x[1] - 1)^2
#     dom = MU.FreeDomain(1)
#
#     (term, prim, mu) = run_muconvexityJuMP(x, poly, dom, use_wsos)
#     @test term == MOI.OPTIMAL
#     @test prim == MOI.FEASIBLE_POINT
#     @test mu ≈ -4 atol = 1e-4 rtol = 1e-4
# end

# function run_muconvexityJuMP_b(; use_wsos::Bool = true)
#     n = 3
#     DP.@polyvar x[1:n]
#     poly = sum(x.^4) - sum(x.^2)
#     dom = MU.FreeDomain(n)
#
#     (term, prim, mu) = run_muconvexityJuMP(x, poly, dom, use_wsos)
#     @test term == MOI.OPTIMAL
#     @test prim == MOI.FEASIBLE_POINT
#     @test mu ≈ -2 atol = 1e-4 rtol = 1e-4
# end
#
# function run_muconvexityJuMP_c(; use_wsos::Bool = true)
#     DP.@polyvar x[1:1]
#     poly = (x[1] + 1)^2 * (x[1] - 1)^2
#     dom = MU.Box([-1.0], [1.0])
#
#     (term, prim, mu) = run_muconvexityJuMP(x, poly, dom, use_wsos)
#     @test term == MOI.OPTIMAL
#     @test prim == MOI.FEASIBLE_POINT
#     @test mu ≈ -4 atol = 1e-4 rtol = 1e-4
# end
#
# function run_muconvexityJuMP_d(; use_wsos::Bool = true)
#     n = 3 # why was this working? the ball had dimension 2
#     DP.@polyvar x[1:n]
#     poly = sum(x.^4) - sum(x.^2)
#     dom = MU.Ball([5.0, 5.0], 1.0)
#
#     (term, prim, mu) = run_muconvexityJuMP(x, poly, dom, use_wsos)
#     @test term == MOI.OPTIMAL
#     @test prim == MOI.FEASIBLE_POINT
#     @test mu ≈ -2 atol = 1e-4 rtol = 1e-4
# end
