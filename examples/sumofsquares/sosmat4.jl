#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

test whether a given polynomial is convex
=#

using Random
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

function run_JuMP_sosmat4_matrix(
    x::Vector,
    H::Matrix,
    use_wsos::Bool,
    is_SOS::Bool,
    )
    if use_wsos
        n = nvariables(x)
        d = div(maximum(maxdegree.(H)), 2)
        dom = Hypatia.FreeDomain(n)

        model = Model(with_optimizer(Hypatia.Optimizer, verbose=true, tolabsopt=1e-9, tolrelopt=1e-9, tolfeas=1e-9))
        (U, pts, P0, _, _) = Hypatia.interpolate(dom, d, sample_factor=20, sample=true)
        mat_wsos_cone = WSOSPolyInterpMatCone(n, U, [P0])
        @constraint(model, [AffExpr(H[i,j](pts[u, :]) * (i == j ? 1.0 : rt2)) for i in 1:n for j in 1:i for u in 1:U] in mat_wsos_cone)
    else
        model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
        @constraint(model, H in PSDCone())
    end

    JuMP.optimize!(model)

    if is_SOS
        @test JuMP.termination_status(model) == MOI.OPTIMAL
        @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    else
        @test JuMP.termination_status(model) == MOI.INFEASIBLE
        @test JuMP.dual_status(model) == MOI.INFEASIBILITY_CERTIFICATE
    end
    return
end

run_JuMP_sosmat4_poly(x::Vector, poly, use_wsos::Bool, is_SOS::Bool) = run_JuMP_sosmat4_matrix(x, differentiate(poly, x, 2), use_wsos, is_SOS)

function run_JuMP_sosmat4_matrix_a(use_wsos::Bool)
    @polyvar x[1:1]
    M = [x[1]+2x[1]^3 1; -x[1]^2+2 3x[1]^2-x[1]+1]
    MM = M'*M

    run_JuMP_sosmat4_matrix(x, MM, use_wsos, true)
    run_JuMP_sosmat4_matrix(x, -MM, use_wsos, false)
end

function run_JuMP_sosmat4_matrix_b(use_wsos::Bool; rseed::Int=1)
    n = 3
    m = 3
    d = 1

    @polyvar x[1:n]
    Z = monomials(x, 0:d)
    Random.seed!(rseed)
    M = [sum(rand() * Z[l] for l in 1:length(Z)) for i in 1:m, j in 1:m]
    MM = M'*M
    MM = 0.5*(MM + MM')

    run_JuMP_sosmat4_matrix(x, MM, use_wsos, true)
    run_JuMP_sosmat4_matrix(x, -MM, use_wsos, false)
end

function run_JuMP_sosmat4_poly_a(use_wsos::Bool)
    @polyvar x[1:1]
    poly = x[1]^4 + 2x[1]^2

    run_JuMP_sosmat4_poly(x, poly, use_wsos, true)
    run_JuMP_sosmat4_poly(x, -poly, use_wsos, false)
end

function run_JuMP_sosmat4_poly_b(use_wsos::Bool)
    @polyvar x[1:2]
    poly = (x[1]+x[2])^4 + (x[1]+x[2])^2

    run_JuMP_sosmat4_poly(x, poly, use_wsos, true)
    run_JuMP_sosmat4_poly(x, -poly, use_wsos, false)
end

# run_JuMP_sosmat4_poly_c(use_wsos::Bool) = (@polyvar x[1:2]; run_JuMP_sosmat4_poly(x, (x[1]*x[2]-x[1]+2x[2]-x[2]^2)^2, use_wsos)) # TODO SOS but not detected?
