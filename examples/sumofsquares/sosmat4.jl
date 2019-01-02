#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

test whether a given polynomial is convex or concave
=#

using JuMP
using MathOptInterface
MOI = MathOptInterface
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using SumOfSquares
using PolyJuMP
using Test

const rt2 = sqrt(2)

function run_JuMP_sosmat4(x, poly, use_wsos::Bool)
    H = differentiate(poly, x, 2)

    if use_wsos
        n = nvariables(x)
        d = div(maximum(maxdegree.(H)), 2)
        dom = Hypatia.FreeDomain(n)

        model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
        (U, pts, P0, _, _) = Hypatia.interpolate(dom, d, sample_factor=20, sample=true)
        mat_wsos_cone = WSOSPolyInterpMatCone(n, U, [P0])
        @constraint(model, [AffExpr(H[i,j](pts[u, :])) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
        @constraint(model, H in PSDCone())
    end

    JuMP.optimize!(model)

    return (JuMP.termination_status(model) == MOI.OPTIMAL)
end

run_JuMP_sosmat4_a(use_wsos::Bool) = (@polyvar x[1:1]; run_JuMP_sosmat4(x, x[1]^4+2x[1]^2, use_wsos))
run_JuMP_sosmat4_b(use_wsos::Bool) = (@polyvar x[1:1]; run_JuMP_sosmat4(x, -x[1]^4-2x[1]^2, use_wsos))
run_JuMP_sosmat4_c(use_wsos::Bool) = (@polyvar x[1:2]; run_JuMP_sosmat4(x, (x[1]*x[2]-x[1]+2x[2]-x[2]^2)^2, use_wsos))
run_JuMP_sosmat4_d(use_wsos::Bool) = (@polyvar x[1:2]; run_JuMP_sosmat4(x, (x[1]+x[2])^4 + (x[1]+x[2])^2, use_wsos))
run_JuMP_sosmat4_e(use_wsos::Bool) = (@polyvar x[1:2]; run_JuMP_sosmat4(x, -(x[1]+x[2])^4 + (x[1]+x[2])^2, use_wsos))
