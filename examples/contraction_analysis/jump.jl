#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

example taken from "Stability and robustness analysis of nonlinear systems via contraction metrics and SOS programming" by Aylward, E.M., Parrilo, P.A. and Slotine, J.J.E.
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const SO = HYP.Solvers
const MO = HYP.Models
const MU = HYP.ModelUtilities

import JuMP
import SumOfSquares
import MathOptInterface
const MOI = MathOptInterface
import PolyJuMP
const PJ = PolyJuMP
import DynamicPolynomials
const DP = DynamicPolynomials
using LinearAlgebra
using Test

const rt2 = sqrt(2)

# dynamics according to the Moore-Greitzer model
function build_dynamics(x)
    dx1dt = -x[2] - 1.5 * x[1]^2 - 0.5 * x[1]^3
    dx2dt = 3 * x[1] - x[2]
    return [dx1dt; dx2dt]
end

function contraction_analysis_common(beta::Float64, deg_M::Int, delta::Float64 = 1e-3)
    n = 2
    dom = MU.FreeDomain(n)

    d_M = div(deg_M + 1, 2)
    (U_M, pts_M, P0_M, _, _) = MU.interpolate(dom, d_M, sample = false)
    lagrange_polys = MU.recover_lagrange_polys(pts_M, 2d_M)

    polyjump_basis = PJ.FixedPolynomialBasis(lagrange_polys)
    x = DP.variables(lagrange_polys[1])
    dynamics = build_dynamics(x)

    model = SumOfSquares.SOSModel(JuMP.with_optimizer(Hypatia.Optimizer, verbose = true, tol_feas = 1e-4, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6))
    JuMP.@variable(model, polys[1:3], PJ.Poly(polyjump_basis))

    dfdx = DP.differentiate(dynamics, x)'
    M = [polys[1] polys[2]; polys[2] polys[3]]
    dMdt = [JuMP.dot(DP.differentiate(M[i, j], x), dynamics) for i in 1:n, j in 1:n]
    Mdfdx = [sum(M[i, k] * dfdx[k, j] for k in 1:n) for i in 1:n, j in 1:n]
    R = Mdfdx + Mdfdx' + dMdt + beta * M
    deg_R = maximum(DP.maxdegree.(R))
    d_R = div(deg_R + 1, 2)
    (U_R, pts_R, P0_R, _, _) = MU.interpolate(dom, d_R, sample = true)

    return (model, M, R, pts_M, pts_R, U_M, U_R, P0_M, P0_R)
end

function contraction_analysis_WSOS(beta::Float64, deg_M::Int; delta::Float64 = 1e-3, use_scalar::Bool = true)
    n = 2
    (model, M, R, pts_M, pts_R, U_M, U_R, P0_M, P0_R) = contraction_analysis_common(beta, deg_M, delta)
    if use_scalar
        DP.@polyvar y[1:n]
        conv_condition_M = y' * M * y
        (naive_U_M, naive_points_M, naive_P0_M, _) = MU.bilinear_terms(U_M, pts_M, P0_M, [], n)
        wsos_cone_M = HYP.WSOSPolyInterpCone(naive_U_M, [naive_P0_M])
        JuMP.@constraint(model, [conv_condition_M(naive_points_M[u, :]) - delta for u in 1:naive_U_M] in wsos_cone_M)

        conv_condition_R = -y' * R * y
        (naive_U_R, naive_points_R, naive_P0_R, _) = MU.bilinear_terms(U_R, pts_R, P0_R, [], n)
        wsos_cone_R = HYP.WSOSPolyInterpCone(naive_U_R, [naive_P0_R])
        JuMP.@constraint(model, [conv_condition_R(naive_points_R[u, :]) - delta for u in 1:naive_U_R] in wsos_cone_R)
    else
        JuMP.@constraint(model, [M[i, j](pts_M[u, :]) * (i == j ? 1.0 : rt2) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_M] in HYP.WSOSPolyInterpMatCone(n, U_M, [P0_M]))
        JuMP.@constraint(model, [-1 * R[i, j](pts_R[u, :]) * (i == j ? 1.0 : rt2) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_R] in HYP.WSOSPolyInterpMatCone(n, U_R, [P0_R]))
    end
    return model
end

function contraction_analysis_PSD(beta::Float64, deg_M::Int; delta::Float64 = 1e-3)
    n = 2
    (model, M, R, _, _, _, _, _, _) = contraction_analysis_common(beta, deg_M, delta)
    JuMP.@constraint(model, M - delta * Matrix{Float64}(I, n, n) in JuMP.PSDCone())
    JuMP.@constraint(model, -R - delta * Matrix{Float64}(I, n, n) in JuMP.PSDCone())
    return model
end

function run_JuMP_contraction_analysis(use_wsos::Bool)
    if use_wsos
        model = contraction_analysis_WSOS(0.79, 4)
    else
        model = contraction_analysis_PSD(0.79, 4)
    end
    JuMP.optimize!(model)
    term_status = JuMP.termination_status(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.OPTIMAL
    @test pr_status == MOI.FEASIBLE_POINT
    @test du_status == MOI.FEASIBLE_POINT

    return nothing
end

run_JuMP_contraction_analysis_PSD() = run_JuMP_contraction_analysis(false)
run_JuMP_contraction_analysis_WSOS() = run_JuMP_contraction_analysis(true)
