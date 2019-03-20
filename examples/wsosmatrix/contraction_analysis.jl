#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

example taken from "Stability and robustness analysis of nonlinear systems via contraction metrics and SOS programming" by Aylward, E.M., Parrilo, P.A. and Slotine, J.J.E.
=#

import Hypatia
import MathOptInterfaceMosek
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

function jet_engine_WSOS(beta::Float64, deg_M::Int)
    n = 2
    dom = MU.FreeDomain(n)
    delta = 1e-3

    # for the matrix M
    d_M = div(deg_M + 1, 2)
    (U_M, pts_M, P0_M, _, _) = MU.interpolate(dom, d_M, sample = false)
    lagrange_polys = MU.recover_lagrange_polys(pts_M, 2d_M)

    # all polynomials in the model will use the Lagrange basis
    polyjump_basis = PJ.FixedPolynomialBasis(lagrange_polys)
    # polyvars we will use throughout
    x = DP.variables(lagrange_polys[1])
    # dynamics according to the Moore-Greitzer model
    dx1dt = -x[2] - 1.5 * x[1]^2 - 0.5 * x[1]^3
    dx2dt = 3 * x[1] - x[2]
    dynamics = [dx1dt; dx2dt]
    # Jacobian
    dfdx = DP.differentiate([dx1dt; dx2dt], x)'

    # for the matrix R
    deg_R = deg_M + maximum(DP.maxdegree.(dfdx))
    d_R = div(deg_R + 1, 2)
    (U_R, pts_R, P0_R, _, _) = MU.interpolate(dom, deg_R, sample = false)

    model = JuMP.Model(JuMP.with_optimizer(Hypatia.Optimizer, verbose = true, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6))
    JuMP.@variable(model, M[i = 1:n, j = 1:i], variable_type = PJ.Poly(polyjump_basis))
    # TODO ask benoit about grabbing coefficients from defined basis as variables, instead of doing this substitution
    JuMP.@constraint(model, [M[i, j](pts_M[u, :]) * (i == j ? 1.0 : rt2) - (i == j ? delta : 0.0)
        for i in 1:n for j in 1:i for u in 1:U_M] in HYP.WSOSPolyInterpMatCone(n, U_M, [P0_M]))
    dMdt(i, j) = JuMP.dot(DP.differentiate(M[i, j], x), dynamics)
    Minds(i, j) = M[max(i, j), min(i, j)]
    Mdfdx(i, j) = sum(Minds(i, k) * dfdx[k, j] for k in 1:n)
    Rmat(i, j) = Mdfdx(j, i) + Mdfdx(i, j) + dMdt(i, j) + beta * M[i, j]
    JuMP.@constraint(model, -[Rmat(i, j)(pts_R[u, :]) * (i == j ? 1.0 : rt2) - (i == j ? delta : 0.0)
        for i in 1:n for j in 1:i for u in 1:U_R] in HYP.WSOSPolyInterpMatCone(n, U_R, [P0_R]))

    JuMP.optimize!(model)

end

function jet_engine_SDP(beta::Float64, deg_M::Int)
    n = 2
    delta = 1e-3
    @polyvar x[1:n]
    dx1dt = -x[2] - 1.5 * x[1]^2 - 0.5 * x[1]^3
    dx2dt = 3 * x[1] - x[2]
    dynamics = [dx1dt; dx2dt]
    # Jacobian
    dfdx = DP.differentiate([dx1dt; dx2dt], x)'

    model = SumOfSquares.SOSModel(with_optimizer(MathOptInterfaceMosek.MosekOptimizer))
    JuMP.@variable(model, polys[1:3], PJ.Poly(DP.monomials(x, 0:deg_M))) # TODO find proper way to create polynomial matrix
    M = [polys[1] polys[2]; polys[2] polys[3]]
    dMdt = [JuMP.dot(DP.differentiate(M[i, j], x), dynamics) for i in 1:n, j in 1:n]
    Mdfdx = [sum(M[i, k] * dfdx[k, j] for k in 1:n) for i in 1:n, j in 1:n]
    Rmat = Mdfdx + Mdfdx' + dMdt + beta * M
    JuMP.@constraint(model, M - delta .* Matrix{Float64}(I, n, n) in JuMP.PSDCone())
    JuMP.@constraint(model, -Rmat - delta .* Matrix{Float64}(I, n, n) in JuMP.PSDCone())
    # JuMP.@constraint(model, M - delta .* Matrix{Float64}(I, n, n) in SumOfSquares.SOSMatrixCone())
    # JuMP.@constraint(model, -Rmat - delta .* Matrix{Float64}(I, n, n) in SumOfSquares.SOSMatrixCone())

    JuMP.optimize!(model)
    JuMP.termination_status(model)

    for i in 1:20
        x = rand(2)
        mcheck = JuMP.value.([M[1, 1](x) M[1, 2](x); M[2, 1](x) M[2, 2](x)])
        @assert isposdef(mcheck)
        rcheck = JuMP.value.([Rmat[1, 1](x) Rmat[1, 2](x); Rmat[1, 2](x) Rmat[2, 2](x)])
        @assert isposdef(-rcheck)
    end

end

function find_beta()
    deg_M = 2
    for beta in [0.0]
        jet_engine_WSOS(beta, deg_M)
    end
end
