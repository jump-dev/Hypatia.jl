#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MU = HYP.ModelUtilities

import MathOptInterface
const MOI = MathOptInterface
import JuMP
# import PolyJuMP
import DynamicPolynomials
const DP = DynamicPolynomials
# import SumOfSquares
using LinearAlgebra
import Random
using Test

const rt2 = sqrt(2)

function JuMP_polysoc_small(; rsoc = false)
    dom = MU.FreeDomain(n)
    DP.@polyvar x
    poly = x^6 + 3.5x^3 - 17x^2 - 6
    d = 6
    (U, pts, P0, _, w) = MU.interpolate(dom, d, sample = false, calc_w = true)
    lagrange_polys = MU.recover_lagrange_polys(pts, 2d)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, max_iters = 400, tol_feas = 1e-5))
    JuMP.@variable(model, f[1:U])
    JuMP.@objective(model, Min, dot(w, f))
    if rsoc
        var1 = f + 0.5 * ones(U)
        var2 = f - 0.5 * ones(U)
        var3 = 2 * [poly(pts[u, :]) for u in 1:U]
        cone = HYP.WSOSPolyInterpSOCCone(3, U, [P0])
        JuMP.@constraint(model, vcat(var1, var2, var3) in cone)
        JuMP.optimize!(model)
        sqr_poly = dot(JuMP.value.(f), lagrange_polys)
        poly2 = poly^2
        for _ in 1:2000
            x = randn() * 10 # need higher feasibility tolerance for crazier numbers
            if !(poly2(x) <= sqr_poly(x))
                @show x, poly2(x), sqr_poly(x)
            end
        end
    else
        cone = HYP.WSOSPolyInterpSOCCone(2, U, [P0])
        JuMP.@constraint(model, vcat(f, [poly(pts[u, :]) for u in 1:U]...) in cone)
        JuMP.optimize!(model)
    end
    @show dot(JuMP.value.(f), lagrange_polys)
    return model
end


function JuMP_polysoc_envelope(; use_scalar = false)
    Random.seed!(1)
    n = 2
    dom = MU.FreeDomain(n)
    DP.@polyvar x[1:n]
    d = 2
    (U, pts, P0, _, w) = MU.interpolate(dom, d, sample = false, calc_w = true)
    lagrange_polys = MU.recover_lagrange_polys(pts, 2d)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, max_iters = 400, tol_feas = 1e-7))
    JuMP.@variable(model, f[1:U])
    JuMP.@objective(model, Min, dot(w, f))

    vec_length = 5
    npoly = vec_length - 1
    LDegs = size(P0, 2)
    polys = P0[:, 1:LDegs] * rand(-9:9, LDegs, npoly)
    rand_polys = [dot(polys[:, i], lagrange_polys) for i in 1:npoly]
    # @show sum(rand_polys.^2)

    fpoly = dot(f, lagrange_polys)
    rand_polys = [dot(polys[:, i], lagrange_polys) for i in 1:npoly]
    DP.@polyvar y[1:vec_length]
    soc_condition = fpoly * sum(y[i]^2 for i in 1:vec_length) + 2 * sum(rand_polys[i - 1] * y[1] * y[i] for i in 2:vec_length)
    DP.@polyvar z[1:vec_length]
    sdp_condition = fpoly * sum(z[i]^2 for i in 1:vec_length) + 2 * sum(rand_polys[i - 1] * z[1] * z[i] for i in 2:vec_length)
    if use_scalar
        # (naive_U, naive_pts, naive_P0, _) = MU.soc_terms(U, pts, P0, [], vec_length)
        # cone = HYP.WSOSPolyInterpCone(naive_U, [naive_P0])
        # JuMP.@constraint(model, [soc_condition(naive_pts[u, :]) for u in 1:naive_U] in cone)
        (naive_U, naive_pts, naive_P0, _) = MU.bilinear_terms(U, pts, P0, [], vec_length)
        wsos_cone = HYP.WSOSPolyInterpCone(naive_U, [naive_P0])
        JuMP.@constraint(model, [sdp_condition(naive_pts[u, :]) for u in 1:naive_U] in wsos_cone)
    else
        cone = HYP.WSOSPolyInterpSOCCone(vec_length, U, [P0])
        JuMP.@constraint(model, vcat(f, [polys[:, i] for i in 1:npoly]...) in cone)
    end
    JuMP.optimize!(model)

    # for _ in 1:20000
    #     rndpt = randn(n + vec_length) * 10
    #     if JuMP.value(soc_condition)(rndpt) <= 0
    #         @show rndpt
    #     end
    # end
    # for _ in 1:20000
    #     rndpt = randn(n + vec_length) * 10
    #     if JuMP.value(sdp_condition)(rndpt) < 0
    #         @show rndpt, JuMP.value(sdp_condition)(rndpt)
    #     end
    #     if (dot(JuMP.value.(f), lagrange_polys)(rndpt[1:n]))^2 - sum(abs2(rp(rndpt[1:n])) for rp in rand_polys)  < 0
    #         @show rndpt, JuMP.value(sdp_condition)(rndpt)
    #     end
    # end

    @show dot(JuMP.value.(f), lagrange_polys)
    @show JuMP.objective_value(model)
    return model
end

# using Plots
# plotlyjs()
# func1(x) = sum(rand_polys.^2)(x)
# func2(x) = dot(JuMP.value.(f), lagrange_polys)(x)^2
# plot(func1, -1, 1)
# plot!(func2, -1, 1)


function JuMP_polysoc_monomial(P, n)
    dom = MU.FreeDomain(n)
    d = div(maximum(DP.maxdegree.(P)) + 1, 2)
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample = false)
    cone = HYP.WSOSPolyInterpSOCCone(length(P), U, [P0])
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, max_iters = 200))
    JuMP.@constraint(model, [P[i](pts[u, :]) for i in 1:length(P) for u in 1:U] in cone)
    return model
end

function simple_feasibility()
    DP.@polyvar x
    for socpoly in [
            [2x^2 + 2, x, x],
            [x^2 + 2, x], [x^2 + 2, x, x],
            [2 * x^4 + 8 * x^2 + 4, x + 2 + (x + 1)^2, x],
            ]
        model = JuMP_polysoc_monomial(socpoly, 1)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.OPTIMAL
        @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    end
end


function simple_infeasibility()
    DP.@polyvar x
    for socpoly in [
        [x, x^2 + x],
        [x, x + 1],
        [x^2, x],
        [x + 2, x],
        [x - 1, x, x],
        ]
        @show socpoly
        model = JuMP_polysoc_monomial(socpoly, 1)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.INFEASIBLE
        @test JuMP.primal_status(model) == MOI.INFEASIBLE_POINT
    end
end


# @testset "everything" begin
#     simple_feasibility()
#     simple_infeasibility()
#
#     Random.seed!(1)
#     for deg in 1:2, n in 1:2, npolys in 1:2
#         println()
#         @show deg, n, npolys
#
#         dom = MU.FreeDomain(n)
#         d = div(deg + 1, 2)
#         (U, pts, P0, _, w) = MU.interpolate(dom, d, sample = false, calc_w = true)
#         lagrange_polys = MU.recover_lagrange_polys(pts, 2d)
#
#         # generate vector of random polys using the Lagrange basis
#         random_coeffs = Random.rand(npolys, U)
#         subpolys = [LinearAlgebra.dot(random_coeffs[i, :], lagrange_polys) for i in 1:npolys]
#         random_vec = [random_coeffs[i, u] for i in 1:npolys for u in 1:U]
#
#         model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, max_iters = 100))
#         JuMP.@variable(model, coeffs[1:U])
#         JuMP.@constraint(model, [coeffs; random_vec...] in HYP.WSOSPolyInterpSOCCone(npolys + 1, U, [P0]))
#         # JuMP.@objective(model, Min, dot(quad_weights, coeffs))
#         JuMP.optimize!(model)
#         upper_bound = LinearAlgebra.dot(JuMP.value.(coeffs), lagrange_polys)
#         @test JuMP.termination_status(model) == MOI.OPTIMAL
#         @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
#
#         for i in 1:50
#             pt = randn(n)
#             @test (upper_bound(pt))^2 >= sum(subpolys.^2)(pt)
#         end
#     end
# end
