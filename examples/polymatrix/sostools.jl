#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

sosmatrixA: example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/simplematrixsos.jl
Example 3.77 and 3.79 of Blekherman, G., Parrilo, P. A., & Thomas, R. R. (Eds.),
Semidefinite optimization and convex algebraic geometry SIAM 2013

sosmatrixB: example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/choi.jl
verifies that a given polynomial matrix is not a Sum-of-Squares matrix
see Choi, M. D., "Positive semidefinite biquadratic forms",
Linear Algebra and its Applications, 1975, 12(2), 95-100

sosmatrixC: example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/sosdemo9.jl
Section 3.9 of SOSTOOLS User's Manual, see https://www.cds.caltech.edu/sostools/
=#

using Test
import Random
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
import SumOfSquares
import PolyJuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

const rt2 = sqrt(2)

function sosmatrixA()
    Random.seed!(1)

    DynamicPolynomials.@polyvar x
    P = [(x^2 - 2x + 2) x; x x^2]
    dom = MU.FreeDomain(1)

    d = div(maximum(DynamicPolynomials.maxdegree.(P)) + 1, 2)
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)
    mat_wsos_cone = HYP.WSOSPolyInterpMatCone(2, U, [P0])

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@constraint(model, [P[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)

    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT

    return
end

function sosmatrixB(use_matrixwsos::Bool, use_dual::Bool)
    Random.seed!(1)

    DynamicPolynomials.@polyvar x y z
    C = [
        (x^2 + 2y^2) (-x * y) (-x * z);
        (-x * y) (y^2 + 2z^2) (-y * z);
        (-x * z) (-y * z) (z^2 + 2x^2);
        ] .* (x * y * z)^0
    dom = MU.FreeDomain(3)
    d = div(maximum(DynamicPolynomials.maxdegree.(C)) + 1, 2)

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))

    if use_matrixwsos
        (U, pts, P0, _, _) = MU.interpolate(dom, d, sample = false)
        mat_wsos_cone = HYP.WSOSPolyInterpMatCone(3, U, [P0], use_dual)
        if use_dual
            JuMP.@variable(model, z[i in 1:3, 1:i, 1:U])
            JuMP.@constraint(model, [z[i, j, u] * (i == j ? 1.0 : rt2) for i in 1:3 for j in 1:i for u in 1:U] in mat_wsos_cone)
            JuMP.@objective(model, Min, sum(z[i, j, u] * C[i, j](pts[u, :]...) * (i == j ? 1.0 : 2.0) for i in 1:3 for j in 1:i for u in 1:U))
        else
            JuMP.@constraint(model, [C[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:3 for j in 1:i for u in 1:U] in mat_wsos_cone)
        end
    else
        if use_dual
            error("dual formulation not implemented for scalar SOS formulation")
        end
        dom2 = MU.add_free_vars(dom)
        (U, pts, P0, _, _) = MU.interpolate(dom2, d + 2, sample_factor = 20, sample = true)
        scalar_wsos_cone = HYP.WSOSPolyInterpCone(U, [P0])
        DynamicPolynomials.@polyvar w[1:3]
        wCw = w' * C * w
        JuMP.@constraint(model, [wCw(pts[u, :]) for u in 1:U] in scalar_wsos_cone)
    end

    JuMP.optimize!(model)
    if use_dual
        @test JuMP.termination_status(model) == MOI.DUAL_INFEASIBLE
        @test JuMP.primal_status(model) == MOI.INFEASIBILITY_CERTIFICATE
    else
        @test JuMP.termination_status(model) == MOI.INFEASIBLE
        @test JuMP.dual_status(model) == MOI.INFEASIBILITY_CERTIFICATE
    end

    return
end

sosmatrixB_scalar() = sosmatrixB(false, false)
sosmatrixB_wsosmat() = sosmatrixB(true, false)
sosmatrixB_wsosmat_dual() = sosmatrixB(true, true)

function sosmatrixC(use_dual::Bool)
    Random.seed!(1)

    DynamicPolynomials.@polyvar x1 x2 x3
    P = [
        (x1^4 + x1^2 * x2^2 + x1^2 * x3^2) (x1 * x2 * x3^2 - x1^3 * x2 - x1 * x2 * (x2^2 + 2 * x3^2));
        (x1 * x2 * x3^2 - x1^3 * x2 - x1 * x2 * (x2^2 + 2 * x3^2)) (x1^2 * x2^2 + x2^2 * x3^2 + (x2^2 + 2 * x3^2)^2);
        ]
    dom = MU.FreeDomain(3)

    d = div(maximum(DynamicPolynomials.maxdegree.(P)) + 1, 2)
    (U, pts, P0, _, _) = MU.interpolate(dom, d, sample = false)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    mat_wsos_cone = HYP.WSOSPolyInterpMatCone(2, U, [P0], use_dual)

    if use_dual
        JuMP.@variable(model, z[i in 1:2, 1:i, 1:U])
        JuMP.@constraint(model, [z[i, j, u] * (i == j ? 1.0 : rt2) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)
        JuMP.@objective(model, Min, sum(z[i, j, u] * P[i, j](pts[u, :]...) * (i == j ? 1.0 : 2.0) for i in 1:2 for j in 1:i for u in 1:U))
    else
        JuMP.@constraint(model, [P[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:2 for j in 1:i for u in 1:U] in mat_wsos_cone)
    end

    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT

    return
end

sosmatrixC_primal() = sosmatrixC(false)
sosmatrixC_dual() = sosmatrixC(true)
