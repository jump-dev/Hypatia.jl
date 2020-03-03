#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

univariate cubic dynamical system
example taken from "Convex computation of the region of attraction of polynomial control systems" by D. Henrion and M. Korda
=#

using LinearAlgebra
using Test
import Random
import JuMP
const MOI = JuMP.MOI
import DynamicPolynomials
const DP = DynamicPolynomials
import SemialgebraicSets
const SAS = SemialgebraicSets
import SumOfSquares
import PolyJuMP
import Hypatia
const MU = Hypatia.ModelUtilities

function regionofattr_JuMP(
    ::Type{T},
    deg::Int,
    use_wsos::Bool, # use wsosinterpnonnegative cone, else PSD formulation
    ) where {T <: Float64} # TODO support generic reals
    DP.@polyvar x
    DP.@polyvar t
    f = x * (x - 0.5) * (x + 0.5) * 100

    model = JuMP.Model()
    JuMP.@variables(model, begin
        v, PolyJuMP.Poly(DP.monomials([x; t], 0:deg))
        w, PolyJuMP.Poly(DP.monomials(x, 0:deg))
    end)
    dvdt = DP.differentiate(v, t) + DP.differentiate(v, x) * f
    diffwv = w - DP.subs(v, t => 0.0) - 1.0
    vT = DP.subs(v, t => 1.0)

    if use_wsos
        dom1 = MU.Box{Float64}([-1.0], [1.0]) # just state
        dom2 = MU.Box{Float64}([-1.0, 0.0], [1.0, 1.0]) # state and time
        dom3 = MU.Box{Float64}([-0.01], [0.01]) # state at the end
        halfdeg = div(deg + 1, 2)
        (U1, pts1, Ps1, quad_weights) = MU.interpolate(dom1, halfdeg, sample = false, calc_w = true)
        (U2, pts2, Ps2, _) = MU.interpolate(dom2, halfdeg, sample = false)
        (U3, pts3, Ps3, _) = MU.interpolate(dom3, halfdeg - 1, sample = false)
        wsos_cone1 = Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U1, Ps1)
        wsos_cone2 = Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U2, Ps2)
        wsos_cone3 = Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(U3, Ps3)

        JuMP.@objective(model, Min, sum(quad_weights[u] * w(pts1[u, :]) for u in 1:U1))
        JuMP.@constraints(model, begin
            [-dvdt(pts2[u, :]) for u in 1:U2] in wsos_cone2
            [diffwv(pts1[u, :]) for u in 1:U1] in wsos_cone1
            [vT(pts3[u, :]) for u in 1:U3] in wsos_cone3
            [w(pts1[u, :]) for u in 1:U1] in wsos_cone1
        end)
    else
        int_box_mon(mon) = prod(1 / (p + 1) - (-1)^(p + 1) / (p + 1) for p in DP.exponents(mon))
        int_box(pol) = sum(DP.coefficient(t) * int_box_mon(t) for t in DP.terms(pol))

        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@objective(model, Min, int_box(w))
        JuMP.@constraint(model, -dvdt >= 0, domain = (SAS.@set -1 <= x  && x <= 1 && 0 <= t && t <= 1))
        JuMP.@constraint(model, diffwv >= 0, domain = (SAS.@set -1 <= x && x <= 1))
        JuMP.@constraint(model, vT >= 0, domain = (SAS.@set -0.01 <= x && x <= 0.01))
        JuMP.@constraint(model, w >= 0, domain = (SAS.@set -1 <= x && x <= 1))
    end

    return (model = model,)
end

function test_regionofattr_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = regionofattr_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

regionofattr_JuMP_fast = [
    (4, true),
    (4, false),
    (6, true),
    (6, false),
    (8, true),
    ]
regionofattr_JuMP_slow = [
    (8, false),
    ]
