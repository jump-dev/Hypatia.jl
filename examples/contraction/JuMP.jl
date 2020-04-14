#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

contraction analysis example adapted from
"Stability and robustness analysis of nonlinear systems via contraction metrics and SOS programming"
Aylward, E.M., Parrilo, P.A. and Slotine, J.J.E
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
import DynamicPolynomials
const DP = DynamicPolynomials
import PolyJuMP
import MultivariateBases: FixedPolynomialBasis
import SumOfSquares

struct ContractionJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    beta::Float64
    M_deg::Int
    delta::Float64
    use_matrixwsos::Bool # use wsos matrix cone, else PSD formulation
    is_feas::Bool
end

example_tests(::Type{ContractionJuMP{Float64}}, ::MinimalInstances) = [
    ((0.85, 2, 1e-3, true, false),),
    ((0.85, 2, 1e-3, false, false),),
    ]
example_tests(::Type{ContractionJuMP{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    relaxed_options = (tol_feas = 1e-6, tol_rel_opt = 1e-5, tol_abs_opt = 1e-5)
    return [
    ((0.77, 4, 1e-3, true, true), nothing, options),
    ((0.77, 4, 1e-3, false, true), nothing, options),
    ((0.85, 4, 1e-3, true, false), nothing, relaxed_options),
    ((0.85, 4, 1e-3, false, false), nothing, options),
    ]
end
example_tests(::Type{ContractionJuMP{Float64}}, ::SlowInstances) = [
    ]

function build(inst::ContractionJuMP{T}) where {T <: Float64} # TODO generic reals
    delta = inst.delta
    n = 2
    dom = ModelUtilities.FreeDomain{Float64}(n)

    M_halfdeg = div(inst.M_deg + 1, 2)
    (U_M, pts_M, Ps_M) = ModelUtilities.interpolate(dom, M_halfdeg)
    lagrange_polys = ModelUtilities.recover_lagrange_polys(pts_M, 2 * M_halfdeg)
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
    R = Mdfdx + Mdfdx' + dMdt + inst.beta * M

    if inst.use_matrixwsos
        deg_R = maximum(DP.maxdegree.(R))
        d_R = div(deg_R + 1, 2)
        (U_R, pts_R, Ps_R) = ModelUtilities.interpolate(dom, d_R)
        M_gap = [M[i, j](pts_M[u, :]) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_M]
        R_gap = [-R[i, j](pts_R[u, :]) - (i == j ? delta : 0.0) for i in 1:n for j in 1:i for u in 1:U_R]
        rt2 = sqrt(2)
        JuMP.@constraint(model, ModelUtilities.vec_to_svec!(M_gap, rt2 = rt2, incr = U_M) in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, U_M, Ps_M))
        JuMP.@constraint(model, ModelUtilities.vec_to_svec!(R_gap, rt2 = rt2, incr = U_R) in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, U_R, Ps_R))
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@constraint(model, M - Matrix(delta * I, n, n) in JuMP.PSDCone())
        JuMP.@constraint(model, -R - Matrix(delta * I, n, n) in JuMP.PSDCone())
    end

    return model
end

function test_extra(inst::ContractionJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == (inst.is_feas ? MOI.OPTIMAL : MOI.INFEASIBLE)
end

return ContractionJuMP
