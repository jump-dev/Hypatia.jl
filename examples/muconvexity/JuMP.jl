#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

find parameter of convexity mu for a given polynomial p(x)
ie the largest mu such that p(x) - mu/2*||x||^2 is convex everywhere on given domain
see https://en.wikipedia.org/wiki/Convex_function#Strongly_convex_functions
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
import DynamicPolynomials
const DP = DynamicPolynomials
import SumOfSquares
import PolyJuMP

struct MuConvexityJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    poly::Symbol
    dom::Symbol
    use_matrixwsos::Bool # use wsosinterpposeideftricone, else PSD formulation
    true_mu::Real # optional true value of parameter for testing only
end

muconvexity_data = Dict(
    :poly1 => (x -> (x[1] + 1)^2 * (x[1] - 1)^2),
    :poly2 => (x -> sum(x .^ 4) - sum(x .^ 2)),
    :dom1 => ModelUtilities.FreeDomain{Float64}(1),
    :dom2 => ModelUtilities.Box{Float64}([-1.0], [1.0]),
    :dom3 => ModelUtilities.FreeDomain{Float64}(3),
    :dom4 => ModelUtilities.Ball{Float64}(ones(2), 5.0),
    )

example_tests(::Type{MuConvexityJuMP{Float64}}, ::MinimalInstances) = [
    ((:poly1, :dom1, true, -4), false),
    ]
example_tests(::Type{MuConvexityJuMP{Float64}}, ::FastInstances) = [
    ((:poly1, :dom2, true, -4), false),
    ((:poly1, :dom1, false, -4), false),
    ((:poly1, :dom2, false, -4), false),
    ((:poly2, :dom3, true, -2), false),
    ((:poly2, :dom4, true, -2), false),
    ((:poly2, :dom3, false, -2), false),
    ((:poly2, :dom4, false, -2), false),
    ]
example_tests(::Type{MuConvexityJuMP{Float64}}, ::SlowInstances) = [
    ]

function build(inst::MuConvexityJuMP{T}) where {T <: Float64} # TODO generic reals
    dom = muconvexity_data[inst.dom]
    n = ModelUtilities.get_dimension(dom)
    DP.@polyvar x[1:n]
    poly = muconvexity_data[inst.poly](x)

    model = JuMP.Model()
    JuMP.@variable(model, mu)
    JuMP.@objective(model, Max, mu)

    convpoly = poly - 0.5 * mu * sum(x.^2)
    H = DP.differentiate(convpoly, x, 2)

    if inst.use_matrixwsos
        d = div(maximum(DP.maxdegree.(H)) + 1, 2)
        (U, pts, Ps, _) = ModelUtilities.interpolate(dom, d)
        mat_wsos_cone = Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, U, Ps)
        H_interp = [H[i, j](x => pts[u, :]) for i in 1:n for j in 1:i for u in 1:U]
        JuMP.@constraint(model, ModelUtilities.vec_to_svec!(H_interp, rt2 = sqrt(2), incr = U) in mat_wsos_cone)
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@constraint(model, H in JuMP.PSDCone(), domain = ModelUtilities.get_domain_inequalities(dom, x))
    end

    return model
end

function test_extra(inst::MuConvexityJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    if JuMP.termination_status(model) == MOI.OPTIMAL && !isnan(inst.true_mu)
        # check objective value is correct
        tol = eps(T)^0.25
        @test JuMP.objective_value(model) ≈ inst.true_mu atol = tol rtol = tol
    end
end

return MuConvexityJuMP
