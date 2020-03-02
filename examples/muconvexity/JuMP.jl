#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find parameter of convexity mu for a given polynomial p(x)
ie the largest mu such that p(x) - mu/2*||x||^2 is convex everywhere on given domain
see https://en.wikipedia.org/wiki/Convex_function#Strongly_convex_functions
=#

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

poly1 = (x -> (x[1] + 1)^2 * (x[1] - 1)^2)
poly2 = (x -> sum(x.^4) - sum(x.^2))
dom1 = MU.FreeDomain{Float64}(1)
dom2 = MU.Box{Float64}([-1.0], [1.0])
dom3 = MU.FreeDomain{Float64}(3)
dom4 = MU.Ball{Float64}(ones(2), 5.0)

function muconvexity_JuMP(
    T::Type{Float64}, # TODO support generic reals
    poly::Symbol,
    dom::Symbol,
    use_matrixwsos::Bool, # use wsosinterpposeideftricone, else PSD formulation
    true_mu::Real = NaN, # optional true value of parameter for testing only
    )
    dom = eval(dom)
    n = MU.get_dimension(dom)
    DP.@polyvar x[1:n]
    poly = eval(poly)(x)

    model = JuMP.Model()
    JuMP.@variable(model, mu)
    JuMP.@objective(model, Max, mu)

    convpoly = poly - 0.5 * mu * sum(x.^2)
    H = DP.differentiate(convpoly, x, 2)

    if use_matrixwsos
        d = div(maximum(DP.maxdegree.(H)) + 1, 2)
        (U, pts, Ps, _) = MU.interpolate(dom, d, sample = true, sample_factor = 100)
        mat_wsos_cone = Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, U, Ps)
        H_interp = [H[i, j](x => pts[u, :]) for i in 1:n for j in 1:i for u in 1:U]
        JuMP.@constraint(model, MU.vec_to_svec!(H_interp, rt2 = sqrt(2), incr = U) in mat_wsos_cone)
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@constraint(model, H in JuMP.PSDCone(), domain = MU.get_domain_inequalities(dom, x))
    end

    return (model = model, mu = mu, true_mu = true_mu)
end

function test_muconvexity_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = muconvexity_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    if !isnan(d.true_mu)
        @test JuMP.value(d.mu) ≈ d.true_mu atol = 1e-4 rtol = 1e-4
    end
    return d.model.moi_backend.optimizer.model.optimizer.result
end

muconvexity_JuMP_fast = [
    (:poly1, :dom1, true, -4),
    (:poly1, :dom2, true, -4),
    (:poly1, :dom1, false, -4),
    (:poly1, :dom2, false, -4),
    (:poly2, :dom3, true, -2),
    (:poly2, :dom4, true, -2),
    (:poly2, :dom3, false, -2),
    (:poly2, :dom4, false, -2),
    ]
muconvexity_JuMP_slow = [
    # TODO
    ]
