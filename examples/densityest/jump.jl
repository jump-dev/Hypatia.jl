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

function build_JuMP_densityest_monomials(
    X::Matrix{Float64},
    deg::Int,
    dom::MU.Domain;
    sample_factor::Int = 100,
    )
    (nobs, dim) = size(X)
    d = div(deg + 1, 2)

    (U, pts, P0, PWts, w) = MU.interpolate(dom, d, sample = true, calc_w = true, sample_factor = sample_factor)

    DynamicPolynomials.@polyvar x[1:dim]
    PX = DynamicPolynomials.monomials(x, 1:deg)
    U = size(pts, 1)

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@variables(model, begin
        z[1:nobs] # log(f(u_i)) at each observation
        f, PolyJuMP.Poly(PX) # probability density function
    end)
    JuMP.@objective(model, Max, sum(z)) # maximize log likelihood
    JuMP.@constraints(model, begin
        sum(w[i] * f(pts[i, :]) for i in 1:U) == 1.0 # integrate to 1
        [f(pts[i, :]) for i in 1:U] in HYP.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
        [i in 1:nobs], vcat(z[i], 1.0, f(X[i, :])) in MOI.ExponentialCone() # hypograph of log
    end)

    return model
end

function build_JuMP_densityest_interp(
    X::Matrix{Float64},
    deg::Int,
    dom::MU.Domain;
    sample_factor::Int = 100,
    )
    (nobs, dim) = size(X)
    d = div(deg + 1, 2)

    (U, pts, P0, PWts, w) = MU.interpolate(dom, d, sample = true, calc_w = true, sample_factor = sample_factor)
    # for point u lagrange(t_u) = V p(t_u) where V is U by U, polynomials in the columns are chebyshev polynomials (for orthogonality)
    # for point i f(x_i) = sum_u lagrnage_u(x_i) = V' l where v holds lagrange_u and l is the variable
    # V_cheb2interp_data = zeros(U, U)
    # V_cheb2interp_data[:, 1] .= 1.0
    #
    # u = MU.calc_u(n, 2d, pts)
    # col = 1
    # for t in 1:2d
    #     for xp in Combinatorics.multiexponents(n, t)
    #         col += 1
    #         if any(isodd, xp)
    #             m[col] = 0.0
    #         else
    #             m[col] = m[1] / prod(1.0 - abs2(xp[j]) for j in 1:n)
    #         end
    #         @. @views V_cheb2interp_data[:, col] = u[1][:, xp[1] + 1]
    #         for j in 2:n
    #             @. @views V_cheb2interp_data[:, col] *= u[j][:, xp[j] + 1]
    #         end
    #     end
    # end
    Q = LinearAlgebra.qr(P0).Q

    DynamicPolynomials.@polyvar x[1:dim]
    PX = DynamicPolynomials.monomials(x, 1:deg)
    U = size(pts, 1)

    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    JuMP.@variables(model, begin
        z[1:nobs] # log(f(u_i)) at each observation
        f # probability density function
    end)
    JuMP.@objective(model, Max, sum(z)) # maximize log likelihood
    JuMP.@constraints(model, begin
        sum(w[i] * f(pts[i, :]) for i in 1:U) == 1.0 # integrate to 1
        [f(pts[i, :]) for i in 1:U] in HYP.WSOSPolyInterpCone(U, [P0, PWts...]) # density nonnegative
        [i in 1:nobs], vcat(z[i], 1.0, f(X[i, :])) in MOI.ExponentialCone() # hypograph of log
    end)

    return model
end

function run_JuMP_densityest(; rseed::Int = 1)
    Random.seed!(rseed)

    nobs = 200
    n = 1
    deg = 4

    X = rand(Distributions.Uniform(-1, 1), nobs, n)
    dom = MU.Box(-ones(n), ones(n))

    model = build_JuMP_densityest(X, deg, dom)
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

    return
end
