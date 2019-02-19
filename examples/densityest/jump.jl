#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

given a sequence of observations X_1,...,X_n with each Xᵢ in Rᵈ,
find a density function f maximizing the log likelihood of the observations
    min -∑ᵢ zᵢ
    -zᵢ + log(f(Xᵢ)) ≥ 0 ∀ i = 1,...,n
    ∫f = 1
    f ≥ 0
==#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const SO = HYP.Solvers
const MO = HYP.Models
const MU = HYP.ModelUtilities

import JuMP
import MathOptInterface
const MOI = MathOptInterface
import PolyJuMP
import MultivariatePolynomials
import DynamicPolynomials
using LinearAlgebra
import Random
import Distributions
using Test

function build_JuMP_densityest(
    model,
    X::Matrix{Float64},
    deg::Int,
    dom::MU.Domain;
    sample_factor::Int = 100,
    use_monomials::Bool = true,
    )

    (nobs, dim) = size(X)
    d = div(deg + 1, 2)
    (U, pts, P0, PWts, w) = MU.interpolate(dom, d, sample = true, calc_w = true, sample_factor = sample_factor)

    JuMP.@variable(model, z[1:nobs])
    JuMP.@objective(model, Max, sum(z))

    if use_monomials
        lagrange_polys = []
        DynamicPolynomials.@polyvar x[1:dim]
        PX = DynamicPolynomials.monomials(x, 0:(2 * d))
        JuMP.@variable(model, f, PolyJuMP.Poly(PX))
        JuMP.@constraints(model, begin
            sum(w[i] * f(pts[i, :]) for i in 1:U) == 1.0 # integrate to 1
            [f(pts[i, :]) for i in 1:U] in HYP.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
            [i in 1:nobs], vcat(z[i], 1.0, f(X[i, :])) in MOI.ExponentialCone() # hypograph of log
        end)
    else
        lagrange_polys = MU.recover_interpolant_polys(pts, 2 * d)
        basis_evals = Matrix{Float64}(undef, nobs, U)
        for i in 1:nobs, j in 1:U
            basis_evals[i, j] = lagrange_polys[j](X[i, :])
        end
        JuMP.@variable(model, f[1:U])
        JuMP.@constraints(model, begin
            dot(w, f) == 1.0 # integrate to 1
            f in HYP.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
            [i in 1:nobs], vcat(z[i], 1.0, dot(f, basis_evals[i, :])) in MOI.ExponentialCone() # hypograph of log
        end)
    end
    return (lagrange_polys, f)
end

function run_JuMP_densityest(; rseed::Int = 1)
    Random.seed!(rseed)

    nobs = 200
    n = 1
    deg = 4
    X = rand(Distributions.Uniform(-1, 1), nobs, n)
    dom = MU.Box(-ones(n), ones(n))

    for use_monomials in [true, false]
        model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
        (interp_bases, coeffs) = build_JuMP_densityest(model, X, deg, dom, use_monomials = use_monomials)
        JuMP.optimize!(model)

        term_status = JuMP.termination_status(model)
        primal_obj = JuMP.objective_value(model)
        dual_obj = JuMP.objective_bound(model)
        pr_status = JuMP.primal_status(model)
        du_status = JuMP.dual_status(model)

        @test term_status == MOI.OPTIMAL
        @test pr_status == MOI.FEASIBLE_POINT
        @test du_status == MOI.FEASIBLE_POINT
        @test primal_obj ≈ dual_obj atol = 1e-4 rtol = 1e-4
    end

    return nothing
end
