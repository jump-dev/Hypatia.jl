# #=
# Copyright 2018, Chris Coey, Lea Kapelevich and contributors
#
# test whether a given matrix has a SOS decomposition,
# and use this procedure to check whether a polynomial is globally convex
#
# TODO probably merge with muconvexity, and pull muconvexity out into its own example
# =#
#
# using Test
# import Random
# import MathOptInterface
# const MOI = MathOptInterface
# import JuMP
# import DynamicPolynomials
# import SumOfSquares
# import PolyJuMP
# import Hypatia
# const HYP = Hypatia
# const MU = HYP.ModelUtilities
#
# const rt2 = sqrt(2)
#
# function convexityJuMP(x::Vector{PolyVar{true}}, H::Array{Polynomial{true,Int64},2}; use_wsos::Bool = true)
#     model = JuMP.model()
#     if use_wsos
#         n = DynamicPolynomials.nvariables(x)
#         d = div(maximum(DynamicPolynomials.maxdegree.(H)) + 1, 2)
#         dom = MU.FreeDomain(n)
#         (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)
#         mat_wsos_cone = HYP.WSOSPolyInterpMatCone(n, U, [P0])
#         JuMP.@constraint(model, [H[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)
#     else
#         PolyJuMP.setpolymodule!(model, SumOfSquares)
#         JuMP.@constraint(model, H in JuMP.PSDCone())
#     end
#     return (model = model,)
# end
#
# function convexityJuMP(x::Vector{PolyVar{true}}, poly::Polynomial{true,Int64}; use_wsos::Bool = true)
#     return convexityJuMP(x, DynamicPolynomials.differentiate(poly, x, 2), use_wsos = use_wsos)
# end
#
#
# function convexityJuMP1()
#     DynamicPolynomials.@polyvar x[1:1]
#     M = [(x[1] + 2x[1]^3) 1; (-x[1]^2 + 2) (3x[1]^2 - x[1] + 1)]
#     MM = M' * M
#     return convexityJuMP(x, MM, use_wsos = true)
# end
#
# function convexityJuMP2()
#     DynamicPolynomials.@polyvar x[1:1]
#     poly = x[1]^4 + 2x[1]^2
#     return convexityJuMP(x, poly, use_wsos = true)
# end
#
# function convexityJuMP3()
#     DynamicPolynomials.@polyvar x[1:2]
#     poly = (x[1] + x[2])^4 + (x[1] + x[2])^2
#     return convexityJuMP(x, poly, use_wsos = true)
# end
#
# function test_convexityJuMP(instance::Tuple{Function,Bool}, options)
#     (instance, is_SOS) = instance()
#     JuMP.optimize!(data.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
#     if is_SOS
#         @test JuMP.termination_status(data.model) == MOI.OPTIMAL
#     else
#         @test JuMP.termination_status(data.model) == MOI.INFEASIBLE
#     end
# end
#
# test_convexityJuMP(; options...) = test_convexityJuMP.([
#     convexityJuMP1,
#     convexityJuMP2,
#     convexityJuMP3,
#     ], options = options)
#
# function run_JuMP_sosmatrix(
#     x::Vector,
#     H::Matrix,
#     use_wsos::Bool,
#     is_SOS::Bool,
#     )
#     Random.seed!(1)
#
#     model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
#
#     if use_wsos
#         n = DynamicPolynomials.nvariables(x)
#         d = div(maximum(DynamicPolynomials.maxdegree.(H)) + 1, 2)
#         dom = MU.FreeDomain(n)
#         (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)
#         mat_wsos_cone = HYP.WSOSPolyInterpMatCone(n, U, [P0])
#         JuMP.@constraint(model, [H[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)
#     else
#         PolyJuMP.setpolymodule!(model, SumOfSquares)
#         JuMP.@constraint(model, H in JuMP.PSDCone())
#     end
#
#     JuMP.optimize!(model)
#     if is_SOS
#         @test JuMP.termination_status(model) == MOI.OPTIMAL
#         @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
#     else
#         @test JuMP.termination_status(model) == MOI.INFEASIBLE
#         @test JuMP.dual_status(model) == MOI.INFEASIBILITY_CERTIFICATE
#     end
#     return
# end
#
# run_JuMP_sosmatrix_poly(x::Vector, poly, use_wsos::Bool, is_SOS::Bool) =
#     run_JuMP_sosmatrix(x, DynamicPolynomials.differentiate(poly, x, 2), use_wsos, is_SOS)
#
# function run_JuMP_sosmatrix_rand(; use_wsos::Bool = true, rseed::Int = 1)
#     Random.seed!(rseed)
#     n = 3
#     m = 3
#     d = 1
#
#     DynamicPolynomials.@polyvar x[1:n]
#     Z = DynamicPolynomials.monomials(x, 0:d)
#     M = [sum(rand() * Z[l] for l in 1:length(Z)) for i in 1:m, j in 1:m]
#     MM = M' * M
#     MM = 0.5 * (MM + MM')
#
#     run_JuMP_sosmatrix(x, MM, use_wsos, true)
#     run_JuMP_sosmatrix(x, -MM, use_wsos, false)
# end
#
# function run_JuMP_sosmatrix_a(; use_wsos::Bool = true)
#     DynamicPolynomials.@polyvar x[1:1]
#     M = [(x[1] + 2x[1]^3) 1; (-x[1]^2 + 2) (3x[1]^2 - x[1] + 1)]
#     MM = M' * M
#
#     run_JuMP_sosmatrix(x, MM, use_wsos, true)
#     run_JuMP_sosmatrix(x, -MM, use_wsos, false)
# end
#
# function run_JuMP_sosmatrix_poly_a(; use_wsos::Bool = true)
#     DynamicPolynomials.@polyvar x[1:1]
#     poly = x[1]^4 + 2x[1]^2
#
#     run_JuMP_sosmatrix_poly(x, poly, use_wsos, true)
#     run_JuMP_sosmatrix_poly(x, -poly, use_wsos, false)
# end
#
# function run_JuMP_sosmatrix_poly_b(; use_wsos::Bool = true)
#     DynamicPolynomials.@polyvar x[1:2]
#     poly = (x[1] + x[2])^4 + (x[1] + x[2])^2
#
#     run_JuMP_sosmatrix_poly(x, poly, use_wsos, true)
#     run_JuMP_sosmatrix_poly(x, -poly, use_wsos, false)
# end
