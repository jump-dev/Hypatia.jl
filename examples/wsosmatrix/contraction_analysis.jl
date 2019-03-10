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


function jet_engine_WSOS()
    n = 2
    dom = MU.FreeDomain(n)
    beta = 0.78
    eps = 1e-5

    # for the matrix M
    deg_M = 4
    d_M = div(deg_M + 1, 2)
    (U_M, pts_M, P0_M, _, _) = MU.interpolate(dom, d_M, sample = false)
    lagrange_polys = MU.recover_lagrange_polys(pts_M, 2d_M)

    polyjump_basis = PJ.FixedPolynomialBasis(lagrange_polys)

    x = DP.variables(lagrange_polys[1])
    dx1dt = -x[2] - 1.5 * x[1]^2 - 0.5 * x[1]^3
    dx2dt = 3 * x[1] - x[2]
    dynamics = [dx1dt; dx2dt]
    dfdx = DP.differentiate([dx1dt; dx2dt], x)

    # for the matrix R
    deg_R = deg_M + maximum(DP.maxdegree.(dfdx))
    d_R = div(deg_R + 1, 2)
    (U_R, pts_R, P0_R, _, _) = MU.interpolate(dom, deg_R, sample = false)

    model = JuMP.Model(JuMP.with_optimizer(Hypatia.Optimizer, verbose = true))
    JuMP.@variable(model, M[i = 1:n, j = 1:i], variable_type = PJ.Poly(polyjump_basis))
    JuMP.@constraint(model, [M[i, j](pts_M[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U_M] in HYP.WSOSPolyInterpMatCone(n, U_M, [P0_M]))
    dMdt(i, j) = JuMP.dot(DP.differentiate(M[i, j], x), dynamics)
    Minds(i, j) = M[max(i, j), min(i, j)]
    dfdxinds(i, j) = dfdx[max(i, j), min(i, j)]
    Mdfdx(i, j) = sum(Minds(i, k) * dfdxinds(k, j) for k in 1:n)
    Rmat(i, j) = Mdfdx(j, i) + Mdfdx(i, j) + dMdt(i, j) + beta * M[i, j]
    JuMP.@constraint(model, [Rmat(i, j)(pts_R[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U_R] in HYP.WSOSPolyInterpMatCone(n, U_R, [P0_R]))

    JuMP.optimize!(model)

end
