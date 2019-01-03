#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/simplematrixsos.jl
Example 3.77 and 3.79 of Blekherman, G., Parrilo, P. A., & Thomas, R. R. (Eds.), Semidefinite optimization and convex algebraic geometry SIAM 2013
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

function run_JuMP_sosmat1(use_matrixwsos::Bool)
    @polyvar x
    P = [x^2-2x+2 x; x x^2]
    d = div(maximum(DynamicPolynomials.maxdegree.(P)), 2)
    dom = Hypatia.FreeDomain(1)

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    
    if use_matrixwsos
        (U, pts, P0, _, _) = Hypatia.interpolate(dom, d, sample_factor=20, sample=true)
        mat_wsos_cone = WSOSPolyInterpMatCone(2, U, [P0])
        @constraint(model, [AffExpr(P[i,j](pts[u, :]) * (i == j ? 1.0 : rt2)) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        error("scalar WSOS implementation not currently working for this example; see issue #163")
        # dom2 = Hypatia.add_free_vars(dom)
        # (U, pts, P0, _, _) = Hypatia.interpolate(dom2, d+2, sample_factor=20, sample=true)
        # scalar_wsos_cone = WSOSPolyInterpCone(U, [P0])
        # @polyvar y[1:2]
        # yPy = y'*P*y
        # @constraint(model, [AffExpr(yPy(pts[u, :])) for u in 1:U] in scalar_wsos_cone)
    end

    JuMP.optimize!(model)

    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    return
end

run_JuMP_sosmat1_scalar() = run_JuMP_sosmat1(false)
run_JuMP_sosmat1_matrix() = run_JuMP_sosmat1(true)
