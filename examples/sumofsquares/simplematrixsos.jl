#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/simplematrixsos.jl
Example 3.77 and 3.79 of
Blekherman, G., Parrilo, P. A., & Thomas, R. R. (Eds.).
Semidefinite optimization and convex algebraic geometry SIAM 2013
=#
using JuMP
using MathOptInterface
MOI = MathOptInterface
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using PolyJuMP
using Test

function run_JuMP_simplemat(use_matrixwsos::Bool)
    @polyvar x
    @polyvar dummy
    # TODO get rid of dummy, unnecessary for matrix case
    P = [x^2-2x+2 x; x x^2] .* dummy^0
    d = div(maximum(DynamicPolynomials.maxdegree.(P)), 2)
    dom = Hypatia.FreeDomain(2)

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    if use_matrixwsos
        (U, pts, P0, _, _) = Hypatia.interpolate(dom, d, sample_factor=10, sample=true)
        mat_wsos_cone = WSOSPolyInterpMatCone(2, U, [P0])
        @constraint(model, [AffExpr(P[i,j](pts[u, :])) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        dom2 = Hypatia.add_free_vars(dom)
        (U, pts, P0, _, _) = Hypatia.interpolate(dom2, d+2, sample_factor=10, sample=true)
        scalar_wsos_cone = WSOSPolyInterpCone(U, [P0])
        @polyvar y[1:2]
        yPy = y'*P*y
        @constraint(model, [AffExpr(yPy(pts[u, :])) for u in 1:U] in scalar_wsos_cone)
    end

    JuMP.optimize!(model)
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    return nothing
end

run_JuMP_scalar_simplemat() = run_JuMP_simplemat(false) # predictor fail
run_JuMP_matrix_simplemat() = run_JuMP_simplemat(true)
