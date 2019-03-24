#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const MU = HYP.ModelUtilities

import MathOptInterface
const MOI = MathOptInterface
import JuMP
import PolyJuMP
import DynamicPolynomials
const DP = DynamicPolynomials
import SumOfSquares
import LinearAlgebra
import Random
using Test

const rt2 = sqrt(2)

function JuMP_polysoc_monomial(P, n)
    dom = MU.FreeDomain(n)
    d = div(maximum(DP.maxdegree.(P)) + 1, 2)
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample = false)
    cone = HYP.WSOSPolyInterpSOCCone(length(P), U, [P0])
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, max_iters = 200))
    JuMP.@constraint(model, [P[i](pts[u, :]) for i in 1:length(P) for u in 1:U] in cone)
    return model
end

function simple_feasibility()
    DP.@polyvar x
    for socpoly in [
            [2x^2 + 2; x; x],
            [x^2 + 2; x], [x^2 + 2; x; x],
            # [2 * x^4 + 8 * x^2 + 4, x + 2 + (x + 1)^2, x] numerically unstable
            ]
        model = JuMP_polysoc_monomial(socpoly, 1)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.OPTIMAL
        @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    end
end


function simple_infeasibility()
    DP.@polyvar x
    for socpoly in [[x; x^2 + x], [x; x + 1], [x^2; x], [x + 2, x], [x - 1, x, x]]
        @show socpoly
        model = JuMP_polysoc_monomial(socpoly, 1)
        JuMP.optimize!(model)
        @test JuMP.termination_status(model) == MOI.INFEASIBLE
        @test JuMP.primal_status(model) == MOI.INFEASIBLE_POINT
    end
end


@testset "everything" begin
    simple_feasibility()
    simple_infeasibility()

    Random.seed!(1234)
    for deg in 1:2, n in 1:2, npolys in 1:2
        println()
        @show deg, n, npolys

        dom = MU.FreeDomain(n)
        d = div(deg + 1, 2)
        (U, pts, P0, _, w) = MU.interpolate(dom, d, sample = false, calc_w = true)
        lagrange_polys = MU.recover_lagrange_polys(pts, 2d)

        # generate vector of random polys using the Lagrange basis
        random_coeffs = Random.rand(npolys, U)
        subpolys = [LinearAlgebra.dot(random_coeffs[i, :], lagrange_polys) for i in 1:npolys]
        random_vec = [random_coeffs[i, u] for i in 1:npolys for u in 1:U]

        model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, max_iters = 100))
        JuMP.@variable(model, coeffs[1:U])
        JuMP.@constraint(model, [coeffs; random_vec...] in HYP.WSOSPolyInterpSOCCone(npolys + 1, U, [P0]))
        # JuMP.@objective(model, Min, dot(quad_weights, coeffs))
        JuMP.optimize!(model)
        upper_bound = LinearAlgebra.dot(JuMP.value.(coeffs), lagrange_polys)
        @test JuMP.termination_status(model) == MOI.OPTIMAL
        @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT

        for i in 1:50
            pt = rand(n)
            @test (upper_bound(pt))^2 >= sum(subpolys.^2)(pt)
        end
    end

end
