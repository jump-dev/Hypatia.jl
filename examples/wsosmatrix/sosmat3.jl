#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/sosdemo9.jl
Section 3.9 of SOSTOOLS User's Manual, see https://www.cds.caltech.edu/sostools/
=#

using JuMP
using MathOptInterface
MOI = MathOptInterface
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using PolyJuMP
using Test

const rt2 = sqrt(2)

function run_JuMP_sosmat3()
    @polyvar x1 x2 x3
    P = [x1^4+x1^2*x2^2+x1^2*x3^2 x1*x2*x3^2-x1^3*x2-x1*x2*(x2^2+2*x3^2); x1*x2*x3^2-x1^3*x2-x1*x2*(x2^2+2*x3^2) x1^2*x2^2+x2^2*x3^2+(x2^2+2*x3^2)^2]
    dom = Hypatia.FreeDomain(3)

    d = div(maximum(DynamicPolynomials.maxdegree.(P)), 2)
    (U, pts, P0, _, _) = Hypatia.interpolate(dom, d, sample_factor=20, sample=true)
    mat_wsos_cone = WSOSPolyInterpMatCone(2, U, [P0])

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true, tolabsopt=1e-6, tolrelopt=1e-6, tolfeas=1e-6))
    @constraint(model, [AffExpr(P[i,j](pts[u, :]) * (i == j ? 1.0 : rt2)) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)

    JuMP.optimize!(model)

    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    return
end
