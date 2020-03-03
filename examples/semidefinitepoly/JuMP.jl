#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

test whether a given matrix has a SOS decomposition,
and use this procedure to check whether a polynomial is globally convex
=#

using LinearAlgebra
using Test
import Random
import JuMP
const MOI = JuMP.MOI
import DynamicPolynomials
const DP = DynamicPolynomials
import SumOfSquares
import PolyJuMP
import Hypatia
const MU = Hypatia.ModelUtilities

include(joinpath(@__DIR__, "data.jl"))

function semidefinitepoly_JuMP(
    T::Type{Float64}, # TODO support generic reals
    x::Vector{<:DP.PolyVar},
    H::Matrix{<:DP.Polynomial},
    is_feas::Bool, # whether model should be primal-dual feasible; only for testing
    use_wsosmatrix::Bool, # use wsosinterppossemideftri cone, else PSD formulation
    use_dual::Bool, # use dual formulation, else primal formulation
    )
    model = JuMP.Model()

    if use_wsosmatrix
        side = size(H, 1)
        halfdeg = div(maximum(DP.maxdegree.(H)) + 1, 2)
        n = DP.nvariables(x)
        dom = MU.FreeDomain{Float64}(n)
        (U, pts, Ps, _) = MU.interpolate(dom, halfdeg, sample_factor = 20, sample = true)
        mat_wsos_cone = Hypatia.WSOSInterpPosSemidefTriCone{Float64}(side, U, Ps, use_dual)

        rt2 = sqrt(2)
        H_svec = [H[i, j](pts[u, :]) for i in 1:side for j in 1:i for u in 1:U]
        MU.vec_to_svec!(H_svec, rt2 = rt2, incr = U)
        if use_dual
            JuMP.@variable(model, z[i in 1:side, 1:i, 1:U])
            z_svec = [1.0 * z[i, j, u] for i in 1:side for j in 1:i for u in 1:U]
            MU.vec_to_svec!(z_svec, rt2 = rt2, incr = U)
            JuMP.@constraint(model, z_svec in mat_wsos_cone)
            JuMP.@objective(model, Min, dot(z_svec, H_svec))
        else
            JuMP.@constraint(model, H_svec in mat_wsos_cone)
        end
    else
        if use_dual
            error("dual formulation not implemented for scalar SOS formulation")
        else
            PolyJuMP.setpolymodule!(model, SumOfSquares)
            JuMP.@constraint(model, H in JuMP.PSDCone())
        end
    end

    return (model = model, is_feas = is_feas)
end

semidefinitepoly_JuMP(
    T::Type{Float64},
    x::Vector{DP.PolyVar{true}},
    poly::DP.Polynomial,
    args...
    ) = semidefinitepoly_JuMP(T, x, DP.differentiate(poly, x, 2), args...)

semidefinitepoly_JuMP(
    T::Type{Float64},
    matpoly::Symbol,
    args...
    ) = semidefinitepoly_JuMP(T, get_semidefinitepoly_data(matpoly)..., args...)

function test_semidefinitepoly_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = semidefinitepoly_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) in (d.is_feas ? (MOI.OPTIMAL,) : (MOI.INFEASIBLE, MOI.DUAL_INFEASIBLE))
    return d.model.moi_backend.optimizer.model.optimizer.result
end

semidefinitepoly_JuMP_fast = [
    (:matpoly1, true, true),
    (:matpoly1, true, false),
    (:matpoly1, false, false),
    (:matpoly2, true, true),
    (:matpoly2, true, false),
    (:matpoly2, false, false),
    (:matpoly3, true, true),
    (:matpoly3, true, false),
    (:matpoly3, false, false),
    (:matpoly4, true, true),
    (:matpoly4, true, false),
    (:matpoly4, false, false),
    (:matpoly5, true, true),
    (:matpoly5, true, false),
    (:matpoly5, false, false),
    (:matpoly6, true, true),
    (:matpoly6, true, false),
    (:matpoly6, false, false),
    (:matpoly7, true, true),
    (:matpoly7, true, false),
    (:matpoly7, false, false),
    ]
semidefinitepoly_JuMP_slow = [
    # TODO
    ]
