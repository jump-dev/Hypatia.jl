#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

contraction analysis example adapted from
"Stability and robustness analysis of nonlinear systems via contraction metrics and SOS programming"
Aylward, E.M., Parrilo, P.A. and Slotine, J.J.E
=#

using LinearAlgebra
using Test
import Random
import JuMP
const MOI = JuMP.MOI
import DynamicPolynomials
const DP = DynamicPolynomials
import PolyJuMP
import MultivariateBases: FixedPolynomialBasis
import SumOfSquares
import Hypatia
const MU = Hypatia.ModelUtilities

function contraction_JuMP(
    ::Type{T},
    beta::Float64,
    M_deg::Int,
    delta::Float64,
    use_matrixwsos::Bool, # use wsos matrix cone, else PSD formulation
    ) where {T <: Float64} # TODO support generic reals
    n = 2
    dom = MU.FreeDomain{Float64}(n)

    M_halfdeg = div(M_deg + 1, 2)
    (U_M, pts_M, Ps_M, _) = MU.interpolate(dom, M_halfdeg, sample = false)
    lagrange_polys = MU.recover_lagrange_polys(pts_M, 2 * M_halfdeg)
    x = DP.variables(lagrange_polys)

    # dynamics according to the Moore-Greitzer model
    dx1dt = -x[2] - 1.5 * x[1]^2 - 0.5 * x[1]^3
    dx2dt = 3 * x[1] - x[2]
    dynamics = [dx1dt; dx2dt]

    model = JuMP.Model()
    JuMP.@variable(model, polys[1:3], PolyJuMP.Poly(FixedPolynomialBasis(lagrange_polys)))

    M = [polys[1] polys[2]; polys[2] polys[3]]
    dMdt = [JuMP.dot(DP.differentiate(M[i, j], x), dynamics) for i in 1:n, j in 1:n]
    dfdx = DP.differentiate(dynamics, x)'
    Mdfdx = [sum(M[i, k] * dfdx[k, j] for k in 1:n) for i in 1:n, j in 1:n]
    R = Mdfdx + Mdfdx' + dMdt + beta * M

    if use_matrixwsos
        deg_R = maximum(DP.maxdegree.(R))
        d_R = div(deg_R + 1, 2)
        (U_R, pts_R, Ps_R, _) = MU.interpolate(dom, d_R, sample = true)
        M_gap = [M[i, j](pts_M[u, :]) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_M]
        R_gap = [-R[i, j](pts_R[u, :]) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_R]
        rt2 = sqrt(2)
        JuMP.@constraint(model, MU.vec_to_svec!(M_gap, rt2 = rt2, incr = U_M) in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, U_M, Ps_M))
        JuMP.@constraint(model, MU.vec_to_svec!(R_gap, rt2 = rt2, incr = U_R) in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, U_R, Ps_R))
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@constraint(model, M - Matrix(delta * I, n, n) in JuMP.PSDCone())
        JuMP.@constraint(model, -R - Matrix(delta * I, n, n) in JuMP.PSDCone())
    end

    return (model = model, is_feas = (beta < 0.81))
end

function test_contraction_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = contraction_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == (d.is_feas ? MOI.OPTIMAL : MOI.INFEASIBLE)
    return d.model.moi_backend.optimizer.model.optimizer.result
end

contraction_JuMP_fast = [
    (0.77, 4, 1e-3, true),
    (0.77, 4, 1e-3, false),
    (0.85, 4, 1e-3, true),
    (0.85, 4, 1e-3, false),
    ]
contraction_JuMP_slow = [
    # TODO
    ]
