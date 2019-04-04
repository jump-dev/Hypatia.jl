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

# dynamics according to the Moore-Greitzer model
function build_dynamics(x)
    dx1dt = -x[2] - 1.5 * x[1]^2 - 0.5 * x[1]^3
    dx2dt = 3 * x[1] - x[2]
    return [dx1dt; dx2dt]
end

function jet_engine_common(beta::Float64, deg_M::Int, delta::Float64 = 1e-3)
    n = 2
    dom = MU.FreeDomain(n)

    # for the matrix M
    d_M = div(deg_M + 1, 2)
    (U_M, pts_M, P0_M, _, _) = MU.interpolate(dom, d_M, sample = false)
    lagrange_polys = MU.recover_lagrange_polys(pts_M, 2d_M)

    polyjump_basis = PJ.FixedPolynomialBasis(lagrange_polys)
    x = DP.variables(lagrange_polys[1])
    dynamics = build_dynamics(x)

    model = SumOfSquares.SOSModel(JuMP.with_optimizer(Hypatia.Optimizer, verbose = true, tol_feas = 1e-4, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6))
    JuMP.@variable(model, polys[1:3], PJ.Poly(polyjump_basis))

    @assert length(polys) == 3
    n = length(x)
    dfdx = DP.differentiate(dynamics, x)'
    M = [polys[1] polys[2]; polys[2] polys[3]]
    dMdt = [JuMP.dot(DP.differentiate(M[i, j], x), dynamics) for i in 1:n, j in 1:n]
    Mdfdx = [sum(M[i, k] * dfdx[k, j] for k in 1:n) for i in 1:n, j in 1:n]
    R = Mdfdx + Mdfdx' + dMdt + beta * M

    deg_R = maximum(DP.maxdegree.(R))
    d_R = div(deg_R + 1, 2)
    (U_R, pts_R, P0_R, _, _) = MU.interpolate(dom, deg_R, sample = false)

    return (model, M, R, pts_M, pts_R, U_M, U_R, P0_M, P0_R)
end

function check_solution(M, R, pts_R, cone)
    for i in 1:2000
        x = randn(2)
        mcheck = JuMP.value.([M[1, 1](x) M[1, 2](x); M[2, 1](x) M[2, 2](x)])
        @assert isposdef(mcheck)
        rcheck = JuMP.value.([R[1, 1](x) R[1, 2](x); R[1, 2](x) R[2, 2](x)])
        if !isposdef(-rcheck)
            @show eigen(-rcheck).values
            @show x
            @show i
            idx = 0
            for i in 1:cone.R, j in 1:i, u in 1:cone.U
                idx += 1
                if i == j
                    cone.point[idx] = -JuMP.value(R[i, j])(pts_R[u, :])
                else
                    cone.point[idx] = -JuMP.value(R[i, j])(pts_R[u, :]) * sqrt(2)
                end
            end
            @show CO.check_in_cone(cone)
        end
        # @assert isposdef(-rcheck)
    end
    @show maximum(PJ.maxdegree.(M))
end

function jet_engine_WSOS(beta::Float64, deg_M::Int; delta::Float64 = 1e-3)
    n = 2
    (model, M, R, pts_M, pts_R, U_M, U_R, P0_M, P0_R) = jet_engine_common(beta, deg_M, delta)
    JuMP.@constraint(model, [M[i, j](pts_M[u, :]) * (i == j ? 1.0 : rt2) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_M] in HYP.WSOSPolyInterpMatCone(n, U_M, [P0_M]))
    JuMP.@constraint(model, [-R[i, j](pts_R[u, :]) * (i == j ? 1.0 : rt2) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_R] in HYP.WSOSPolyInterpMatCone(n, U_R, [P0_R]))
    JuMP.optimize!(model)
    dummy_cone = CO.WSOSPolyInterpMat(2, U_R, [P0_R], false)
    check_solution(M, R, pts_R, dummy_cone)
    return model
end

function jet_engine_SDP(beta::Float64, deg_M::Int; delta::Float64 = 1e-3)
    n = 2
    (model, M, R, pts_M, pts_R, U_M, U_R, P0_M, P0_R) = jet_engine_common(beta, deg_M, delta)
    JuMP.@constraint(model, M - delta * Matrix{Float64}(I, n, n) in JuMP.PSDCone())
    JuMP.@constraint(model, -R - delta * Matrix{Float64}(I, n, n) in JuMP.PSDCone())
    JuMP.optimize!(model)
    @show JuMP.termination_status(model)
    # check_solution(M, R)
    return model
end

function find_beta()
    deg_M = 2
    for beta in [0.1]
        jet_engine_WSOS(beta, deg_M)
    end
end
