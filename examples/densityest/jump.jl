#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
Given a sequence of observations X₁,...,Xₙ with each Xᵢ in Rᵈ, find a density function f maximizing the log likelihood of the observations.
    minimize -∑ᵢzᵢ
    subject to -zᵢ + log(f(Xᵢ)) ≥ 0 ∀ i = 1,...,n
    ∫f = 1
    f ≥ 0
where
    - zᵢ is in the hypograph of the log function of f
==#
using LinearAlgebra
using Random
using Distributions
using JuMP
using MathOptInterface
MOI = MathOptInterface
using PolyJuMP
using MultivariatePolynomials
using DynamicPolynomials
using Hypatia
using Test

Random.seed!(1234)

function build_JuMP_densityest(
    X::Matrix{Float64},
    deg::Int,
    dom::Hypatia.InterpDomain;
    pts_factor::Int = 10,
    )

    (nobs, dim) = size(X)
    d = div(deg, 2)

    (L, U, pts, P0, PWts, quad_w) = Hypatia.interp_sample(dom, dim, d, pts_factor=pts_factor, calc_w=true)

    @polyvar x[1:dim]
    PX = PolyJuMP.monomials(x, 1:deg)
    U = size(pts, 1)

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true, tolrelopt=1e-5, tolabsopt=1e-2))

    @variables(model, begin
        # log(f(u_i)) at each observation
        z[1:nobs]
        # probability density function
        f, PolyJuMP.Poly(PX)
    end)

    @constraints(model, begin
        # integrate to 1
        sum(quad_w[i] * f(pts[i,:]) for i in 1:U) == 1.0
        # density must be nonnegative
        [f(pts[i,:]) for i in 1:U] in Hypatia.WSOSPolyInterpCone(U, [P0, PWts...])
        # define hypograph of log function variable
        [i in 1:nobs], vcat(z[i], 1.0, f(X[i,:])) in MOI.ExponentialCone()
    end)

    # maximize log likelihood
    @objective(model, Max, sum(z))

    return model
end

function run_JuMP_densityest()
    nobs = 900
    n = 2
    dist = Truncated(Normal(), -1, 1)
    X = rand(dist, nobs, n)
    dom = Hypatia.Box(-ones(n), ones(n))
    deg = 4

    model = build_JuMP_densityest(X, deg, dom)
    JuMP.optimize!(model)

    term_status = JuMP.termination_status(model)
    pobj = JuMP.objective_value(model)
    dobj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)

    @test term_status == MOI.Success
    @test pr_status == MOI.FeasiblePoint
    @test du_status == MOI.FeasiblePoint
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4
    @test pobj ≈ truemin atol=1e-4 rtol=1e-4

    return nothing
end

# run_JuMP_densityest()
