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
import MultivariatePolynomials
import DynamicPolynomials
import SumOfSquares
import PolyJuMP
using Test
import Random
import LinearAlgebra

function run_transformed_wsospoly(use_dual::Bool)
    Random.seed!(1)

    DynamicPolynomials.@polyvar x
    poly = (x + 1)^2
    dom = MU.FreeDomain(1)

    d = 1
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)
    V = [pts[i]^(2-j) for i in 1:U, j in 0:2]
    transform = [V]
    # transform = [Matrix{Float64}(LinearAlgebra.I, U, U) * 5.0] #rand(1, U)

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))

    if use_dual
        mono_dual_cone = HYP.MonotonicPolyCone(U, [P0], [inv(tr') for tr in transform], use_dual)
        JuMP.@variable(model, y[1:U])
        JuMP.@constraint(model, y in mono_dual_cone)
        JuMP.@objective(model, Min, sum(y[u] * poly(pts[u, :]...) for u in 1:U))
        # JuMP.@objective(model, Min, JuMP.dot(y, [1; 2; 1]))
        # ===
        # JuMP.@variable(model, y[1:U])
        # JuMP.@constraint(model, y in HYP.WSOSPolyInterpCone(U, [P0], use_dual))
        # JuMP.@objective(model, Min, JuMP.dot(y, V * [1; 2; 1]))
    else
        mono_primal_cone = HYP.MonotonicPolyCone(U, [P0], [inv(tr') for tr in transform], use_dual)
        JuMP.@constraint(model, JuMP.AffExpr.([1; 2; 1]) in mono_primal_cone)
        # JuMP.@constraint(model, JuMP.AffExpr.(V * [1; 2; 1]) in HYP.WSOSPolyInterpCone(U, [P0], use_dual))

        # JuMP.@variable(model, A[1:3, 1:3])
        # JuMP.@constraint(model, A * [1; 2; 1] in HYP.WSOSPolyInterpCone(U, [P0], use_dual))
    end

    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    return model
end

function run_transformed_dualnamedpoly()
    dom = MU.FreeDomain(1)
    d = 1
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)

    # ==
    V = [pts[i]^(j) for i in 1:U, j in 0:2]
    transform = [V]
    # transform = [Matrix{Float64}(LinearAlgebra.I, 3, 3)]
    mono_dual_cone = HYP.MonotonicPolyCone(U, [P0], [inv(tr') for tr in transform], true)
    # mono_dual_cone = HYP.MonotonicPolyCone(U, [P0], transform, true)

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@variable(model, x[1:3])
    JuMP.@constraint(model, sum(x) == 1)
    JuMP.@constraint(model, x in mono_dual_cone)
    JuMP.@objective(model, Min, x[1] + x[3])
    JuMP.optimize!(model)
    @show JuMP.primal_status(model)
    return model, x

    # ==
    # V = [pts[i]^(j) for i in 1:U, j in 0:2]
    # # cone = HYP.WSOSPolyInterpCone(U, [P0], true)
    # cone = HYP.MonotonicPolyCone(U, [P0], [Matrix{Float64}(LinearAlgebra.I, 3, 3)], true)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # JuMP.@variable(model, x[1:3])
    # JuMP.@constraint(model, sum(x) == 1)
    # JuMP.@constraint(model, x in cone)
    # JuMP.@objective(model, Min, JuMP.dot(V * [2; 0; 1], x))
    # JuMP.optimize!(model)
    # @show JuMP.primal_status(model)
    # return model, x
end

# model = run_transformed_wsospoly(true)
# model = run_transformed_wsospoly(false)
model, x = run_transformed_dualnamedpoly()

# function build_JuMP_namedpoly_WSOS(
#     x,
#     f,
#     dom::MU.Domain;
#     d::Int = div(DynamicPolynomials.maxdegree(f) + 1, 2),
#     sample::Bool = true,
#     primal_wsos::Bool = true,
#     rseed::Int = 1,
#     )
#     Random.seed!(rseed)
#
#     DynamicPolynomials.@polyvar x[1:2]
#     f = 1-48x[1]^2*x[2]^2+64x[1]^2*x[2]^4+64x[1]^4*x[2]^2
#     dom = MU.Box(-ones(2), ones(2))
#     truemin = 0
#
#     (U, pts, P0, PWts, _) = MU.interpolate(dom, d, sample = sample, sample_factor = 100)
#
#     # build JuMP model
#     model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, tol_feas = 1e-9, tol_rel_opt = 1e-8, tol_abs_opt = 1e-8))
#     if primal_wsos
#         JuMP.@variable(model, a)
#         JuMP.@objective(model, Max, a)
#         JuMP.@constraint(model, [f(pts[j, :]) - a for j in 1:U] in HYP.WSOSPolyInterpCone(U, [P0, PWts...], false))
#     else
#         JuMP.@variable(model, x[1:U])
#         JuMP.@objective(model, Min, sum(x[j] * f(pts[j, :]...) for j in 1:U))
#         JuMP.@constraint(model, sum(x) == 1.0)
#         JuMP.@constraint(model, x in HYP.WSOSPolyInterpCone(U, [P0, PWts...], true))
#     end
#
#     return model
# end

# function run_JuMP_sosmat1()
#     Random.seed!(1)
#
#     DynamicPolynomials.@polyvar x
#     poly0 = x
#     # poly1 = DynamicPolynomials.differentiate(poly0)
#     dom = MU.FreeDomain(1)
#
#     d = 3
#     (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)
#     mat_wsos_cone = HYP.WSOSPolyInterpMatCone(2, U, [P0])
#     V = [pts[i]^j for i in 1:U, j in 1:U]
#
#     model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
#     JuMP.@constraint(model, [P[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)
#
#     JuMP.optimize!(model)
#     @test JuMP.termination_status(model) == MOI.OPTIMAL
#     @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
#     return
# end
