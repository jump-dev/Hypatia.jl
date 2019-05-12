#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

example taken from
"Stability and robustness analysis of nonlinear systems via contraction metrics and SOS programming"
Aylward, E.M., Parrilo, P.A. and Slotine, J.J.E
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

function contraction_JuMP(
    beta::Float64,
    deg_M::Int,
    delta::Float64 = 1e-3;
    use_wsos::Bool = true,
    use_dense::Bool = true,
    )
    n = 2
    dom = MU.FreeDomain(n)

    d_M = div(deg_M + 1, 2)
    (U_M, pts_M, P0_M, _, _) = MU.interpolate(dom, d_M, sample = false)
    lagrange_polys = MU.recover_lagrange_polys(pts_M, 2d_M)

    polyjump_basis = PJ.FixedPolynomialBasis(lagrange_polys)
    x = DP.variables(lagrange_polys[1])

    # dynamics according to the Moore-Greitzer model
    dx1dt = -x[2] - 1.5 * x[1]^2 - 0.5 * x[1]^3
    dx2dt = 3 * x[1] - x[2]
    dynamics = [dx1dt; dx2dt]

    model = JuMP.Model(JuMP.with_optimizer(Hypatia.Optimizer, verbose = true, tol_feas = 1e-4, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, use_dense = use_dense))
    JuMP.@variable(model, polys[1:3], PJ.Poly(polyjump_basis))

    M = [polys[1] polys[2]; polys[2] polys[3]]
    dMdt = [JuMP.dot(DP.differentiate(M[i, j], x), dynamics) for i in 1:n, j in 1:n]
    dfdx = DP.differentiate(dynamics, x)'
    Mdfdx = [sum(M[i, k] * dfdx[k, j] for k in 1:n) for i in 1:n, j in 1:n]
    R = Mdfdx + Mdfdx' + dMdt + beta * M

    if use_wsos
        deg_R = maximum(DP.maxdegree.(R))
        d_R = div(deg_R + 1, 2)
        (U_R, pts_R, P0_R, _, _) = MU.interpolate(dom, d_R, sample = true)
        JuMP.@constraint(model, [M[i, j](pts_M[u, :]) * (i == j ? 1.0 : rt2) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_M] in HYP.WSOSPolyInterpMatCone(n, U_M, [P0_M]))
        JuMP.@constraint(model, [-R[i, j](pts_R[u, :]) * (i == j ? 1.0 : rt2) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_R] in HYP.WSOSPolyInterpMatCone(n, U_R, [P0_R]))
    else
        PJ.setpolymodule!(model, SumOfSquares)
        JuMP.@constraint(model, M - Matrix(delta * I, n, n) in JuMP.PSDCone())
        JuMP.@constraint(model, -R - Matrix(delta * I, n, n) in JuMP.PSDCone())
    end

    return model
end

contraction1_JuMP(; use_dense::Bool = true) = contraction_JuMP(0.77, 4, 1e-3, use_wsos = true, use_dense = use_dense)
contraction2_JuMP(; use_dense::Bool = true) = contraction_JuMP(0.77, 4, 1e-3, use_wsos = false, use_dense = use_dense)

function test_contraction_JuMP(instance::Function)
    model = instance()
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    @test JuMP.dual_status(model) == MOI.FEASIBLE_POINT
    return
end

test_contraction_JuMP_many() = test_contraction_JuMP.([contraction1_JuMP, contraction2_JuMP])
