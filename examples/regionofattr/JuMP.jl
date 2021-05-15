#=
univariate cubic dynamical system
example taken from Section 9.1 of
"Convex computation of the region of attraction of polynomial control systems"
by D. Henrion and M. Korda
=#

import DynamicPolynomials
const DP = DynamicPolynomials
import SemialgebraicSets
const SAS = SemialgebraicSets
import SumOfSquares
import PolyJuMP

struct RegionOfAttrJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    deg::Int
    use_wsos::Bool # use wsosinterpnonnegative cone, else PSD formulation
end

function build(inst::RegionOfAttrJuMP{T}) where {T <: Float64}
    deg = inst.deg
    DP.@polyvar x
    DP.@polyvar t
    f = x * (x - 0.5) * (x + 0.5) * 100

    model = JuMP.Model()
    JuMP.@variables(model, begin
        v, PolyJuMP.Poly(DP.monomials(vcat(x, t), 0:deg))
        w, PolyJuMP.Poly(DP.monomials(x, 0:deg))
    end)
    dvdt = DP.differentiate(v, t) + DP.differentiate(v, x) * f
    diffwv = w - DP.subs(v, t => 0.0) - 1.0
    vT = DP.subs(v, t => 1.0)

    if inst.use_wsos
        dom1 = PolyUtils.BoxDomain{T}([-1.0], [1.0]) # just state
        dom2 = PolyUtils.BoxDomain{T}([-1.0, 0.0], [1.0, 1.0]) # state and time
        dom3 = PolyUtils.BoxDomain{T}([-0.01], [0.01]) # state at the end
        halfdeg = div(deg + 1, 2)
        (U1, pts1, Ps1, _, quad_weights) =
            PolyUtils.interpolate(dom1, halfdeg, get_quadr = true)
        (U2, pts2, Ps2) = PolyUtils.interpolate(dom2, halfdeg)
        (U3, pts3, Ps3) = PolyUtils.interpolate(dom3, halfdeg - 1)
        wsos_cone1 = Hypatia.WSOSInterpNonnegativeCone{T, T}(U1, Ps1)
        wsos_cone2 = Hypatia.WSOSInterpNonnegativeCone{T, T}(U2, Ps2)
        wsos_cone3 = Hypatia.WSOSInterpNonnegativeCone{T, T}(U3, Ps3)

        JuMP.@objective(model, Min, sum(quad_weights[u] * w(pts1[u, :])
            for u in 1:U1))
        JuMP.@constraints(model, begin
            [-dvdt(pts2[u, :]) for u in 1:U2] in wsos_cone2
            [diffwv(pts1[u, :]) for u in 1:U1] in wsos_cone1
            [vT(pts3[u, :]) for u in 1:U3] in wsos_cone3
            [w(pts1[u, :]) for u in 1:U1] in wsos_cone1
        end)
    else
        int_box_mon(mon) = prod(1 / (p + 1) - (-1)^(p + 1) / (p + 1)
            for p in DP.exponents(mon))
        int_box(pol) = sum(DP.coefficient(t) * int_box_mon(t)
            for t in DP.terms(pol))

        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@objective(model, Min, int_box(w))

        JuMP.@constraint(model, -dvdt >= 0, domain =
            (SAS.@set -1 <= x  && x <= 1 && 0 <= t && t <= 1))
        JuMP.@constraint(model, diffwv >= 0, domain =
            (SAS.@set -1 <= x && x <= 1))
        JuMP.@constraint(model, vT >= 0, domain =
            (SAS.@set -0.01 <= x && x <= 0.01))
        JuMP.@constraint(model, w >= 0, domain =
            (SAS.@set -1 <= x && x <= 1))
    end

    return model
end
