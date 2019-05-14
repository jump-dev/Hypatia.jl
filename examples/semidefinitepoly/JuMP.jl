#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

test whether a given matrix has a SOS decomposition,
and use this procedure to check whether a polynomial is globally convex

convexityJuMP5: example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/simplematrixsos.jl
Example 3.77 and 3.79 of Blekherman, G., Parrilo, P. A., & Thomas, R. R. (Eds.),
Semidefinite optimization and convex algebraic geometry SIAM 2013

convexityJuMP6: example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/choi.jl
verifies that a given polynomial matrix is not a Sum-of-Squares matrix
see Choi, M. D., "Positive semidefinite biquadratic forms",
Linear Algebra and its Applications, 1975, 12(2), 95-100

convexityJuMP7: example modified from https://github.com/JuliaOpt/SumOfSquares.jl/blob/master/test/sosdemo9.jl
Section 3.9 of SOSTOOLS User's Manual, see https://www.cds.caltech.edu/sostools/
=#

using Test
import Random
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import DynamicPolynomials
const DP = DynamicPolynomials
import SumOfSquares
import PolyJuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

const rt2 = sqrt(2)

function convexityJuMP(x::Vector{DP.PolyVar{true}}, H::Array{DP.Polynomial{true,Int64},2}; use_wsos::Bool = true)
    model = JuMP.Model()
    if use_wsos
        n = DynamicPolynomials.nvariables(x)
        d = div(maximum(DynamicPolynomials.maxdegree.(H)) + 1, 2)
        dom = MU.FreeDomain(n)
        (U, pts, P0, _, _) = MU.interpolate(dom, d, sample_factor = 20, sample = true)
        mat_wsos_cone = HYP.WSOSPolyInterpMatCone(n, U, [P0])
        JuMP.@constraint(model, [H[i, j](pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)
        JuMP.@constraint(model, H in JuMP.PSDCone())
    end
    return (model = model,)
end

function convexityJuMP(x::Vector{DP.PolyVar{true}}, poly::DP.Polynomial{true,Int64}; use_wsos::Bool = true)
    return convexityJuMP(x, DynamicPolynomials.differentiate(poly, x, 2), use_wsos = use_wsos)
end

function convexityJuMP1()
    DynamicPolynomials.@polyvar x[1:1]
    M = [(x[1] + 2x[1]^3) 1; (-x[1]^2 + 2) (3x[1]^2 - x[1] + 1)]
    MM = M' * M
    return convexityJuMP(x, MM, use_wsos = true)
end

function convexityJuMP2()
    DynamicPolynomials.@polyvar x[1:1]
    poly = x[1]^4 + 2x[1]^2
    return convexityJuMP(x, poly, use_wsos = true)
end

function convexityJuMP3()
    DynamicPolynomials.@polyvar x[1:2]
    poly = (x[1] + x[2])^4 + (x[1] + x[2])^2
    return convexityJuMP(x, poly, use_wsos = true)
end

function convexityJuMP4()
    Random.seed!(1234)
    n = 3
    m = 3
    d = 1
    DynamicPolynomials.@polyvar x[1:n]
    Z = DynamicPolynomials.monomials(x, 0:d)
    M = [sum(rand() * Z[l] for l in 1:length(Z)) for i in 1:m, j in 1:m]
    MM = M' * M
    MM = 0.5 * (MM + MM')
    return convexityJuMP(x, MM, use_wsos = true)
end

# SOSTOOLS examples
function convexityJuMP5()
    DynamicPolynomials.@polyvar x[1:1]
    P = [(x^2 - 2x + 2) x; x x^2]
    return convexityJuMP(x, P, use_wsos = true)
end

function convexityJuMP6()
    DynamicPolynomials.@polyvar x y z
    P = [
        (x^2 + 2y^2) (-x * y) (-x * z);
        (-x * y) (y^2 + 2z^2) (-y * z);
        (-x * z) (-y * z) (z^2 + 2x^2);
        ] .* (x * y * z)^0
    return convexityJuMP([x; y; z], P, use_wsos = true)
end

function convexityJuMP7()
    DynamicPolynomials.@polyvar x1 x2 x3
    P = [
        (x1^4 + x1^2 * x2^2 + x1^2 * x3^2) (x1 * x2 * x3^2 - x1^3 * x2 - x1 * x2 * (x2^2 + 2 * x3^2));
        (x1 * x2 * x3^2 - x1^3 * x2 - x1 * x2 * (x2^2 + 2 * x3^2)) (x1^2 * x2^2 + x2^2 * x3^2 + (x2^2 + 2 * x3^2)^2);
        ]
    return convexityJuMP([x1; x2; x3], P, use_wsos = true)
end

function test_convexityJuMP(instance::Tuple{Function,Bool}; options, rseed::Int = 1)
    Random.seed!(1)
    (instance, is_SOS) = instance
    (model,) = instance()
    JuMP.optimize!(model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    if is_SOS
        @test JuMP.termination_status(model) == MOI.OPTIMAL
    else
        @test JuMP.termination_status(model) == MOI.INFEASIBLE
    end
end

test_convexityJuMP(; options...) = test_convexityJuMP.([
    (convexityJuMP2, true),
    (convexityJuMP1, true),
    (convexityJuMP3, true),
    (convexityJuMP4, true),
    (convexityJuMP5, true),
    (convexityJuMP6, false),
    (convexityJuMP7, true),
    ], options = options)
