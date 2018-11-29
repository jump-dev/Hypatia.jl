#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
Given data (xᵢ, yᵢ), find a polynomial p to solve
    minimize ∑ᵢℓ(p(xᵢ), yᵢ)
    subject to ρⱼ × dᵏp/dtⱼᵏ ≥ 0 ∀ t ∈ D
where
    - dᵏp/dtⱼᵏ is the kᵗʰ derivative of p in direction j,
    - ρⱼ determines the desired sign of the derivative,
    - D is a domain such as a box or an ellipsoid,
    - ℓ is a convex loss function.
See e.g. Chapter 8 of thesis by G. Hall (2018).
=#

using LinearAlgebra
using Random
using Distributions
using MathOptInterface
MOI = MathOptInterface
using JuMP
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using SemialgebraicSets
using SumOfSquares
using PolyJuMP
using Test

function generateregrdata(
    func::Function,
    xmin::Float64,
    xmax::Float64,
    n::Int,
    npoints::Int;
    signal_ratio::Float64 = 1.0,
    rseed::Int = 1,
    )
    @assert 0.0 <= signal_ratio < Inf
    Random.seed!(rseed)
    X = rand(Uniform(xmin, xmax), npoints, n)
    y = [func(X[p,:]) for p in 1:npoints]

    if !iszero(signal_ratio)
        noise = rand(Normal(), npoints)
        noise .*= norm(y)/sqrt(signal_ratio)/norm(noise)
        y .+= noise
    end

    return (X, y)
end

function build_shapeconregr_PSD(
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    mono_dom::Hypatia.InterpDomain,
    conv_dom::Hypatia.InterpDomain,
    mono_profile::Vector{Float64};
    use_leastsqobj::Bool = false,
    )
    (npoints, n) = size(X)
    @polyvar x[1:n]
    mono_bss = Hypatia.Hypatia.get_bss(mono_dom, x)
    conv_bss = Hypatia.Hypatia.get_bss(conv_dom, x)

    model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, p, PolyJuMP.Poly(monomials(x, 0:r)))

    dp = [DynamicPolynomials.differentiate(p, x[i]) for i in 1:n]
    @constraint(model, [j in 1:n], mono_profile[j] * dp[j] >= 0, domain=mono_bss)

    # TODO think about what it means if wsos polynomials have degree > 2
    Hp = [DynamicPolynomials.differentiate(dp[i], x[j]) for i in 1:n, j in 1:n]
    @SDconstraint(model, Hp >= 0, domain=conv_bss)

    if use_leastsqobj
        @variable(model, z)
        @constraint(model, [z, [y[i] - p(X[i, :]) for i in 1:npoints]...] in MOI.SecondOrderCone(1+npoints))
        @objective(model, Min, z)
     else
        @variable(model, z[1:npoints])
        @objective(model, Min, sum(z))
        @constraints(model, begin
            [i in 1:npoints], z[i] >= y[i] - p(X[i, :])
            [i in 1:npoints], z[i] >= -y[i] + p(X[i, :])
        end)
    end

    return (model, p)
end

function doubledomain!(conv_dom::Hypatia.Box)
    append!(conv_dom.l, conv_dom.l)
    append!(conv_dom.u, conv_dom.u)
    return conv_dom
end

function build_shapeconregr_WSOS(
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    mono_dom::Hypatia.InterpDomain,
    conv_dom::Hypatia.InterpDomain,
    mono_profile::Vector{Float64};
    ortho_wts::Bool = true,
    use_leastsqobj::Bool = false,
    )
    @assert mod(r, 2) == 1
    d = div(r-1, 2)
    (npoints, n) = size(X)
    @polyvar x[1:n]
    @polyvar w[1:n]

    doubledomain!(conv_dom)
    (_, mono_U, mono_pts, _, mono_P, mono_PWts, _) = Hypatia.interp_sample(mono_dom, n, d, calc_w=false, ortho_wts=true, pts_factor=100n)
    (_, conv_U, conv_pts, _, conv_P, conv_PWts, _) = Hypatia.interp_sample(conv_dom, 2n, d+1, calc_w=false, ortho_wts=true, pts_factor=100n) # TODO think about if it's ok to go up to d+1
    mono_wsos_cone = WSOSPolyInterpCone(mono_U, [mono_P, mono_PWts...])
    conv_wsos_cone = WSOSPolyInterpCone(conv_U, [conv_P, conv_PWts...])

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variable(model, p, PolyJuMP.Poly(monomials(x, 0:r)))

    # relate coefficients for monotonicity
    dp = [DynamicPolynomials.differentiate(p, x[j]) for j in 1:n]

    # relate coefficients for convexity
    Hp = [DynamicPolynomials.differentiate(dp[i], x[j]) for i in 1:n, j in 1:n]
    conv_condition = w'*Hp*w
    @constraints(model, begin
        [j in 1:n], [mono_profile[j] * dp[j](mono_pts[i, :]) for i in 1:mono_U] in mono_wsos_cone
        [conv_condition(conv_pts[i, :]) for i in 1:conv_U] in conv_wsos_cone
    end)

    if use_leastsqobj
        @variable(model, z)
        @constraint(model, [z, [y[i] - p(X[i, :]) for i in 1:npoints]...] in MOI.SecondOrderCone(1+npoints))
        @objective(model, Min, z)
     else
        @variable(model, z[1:npoints])
        @objective(model, Min, sum(z))
        @constraints(model, begin
            [i in 1:npoints], z[i] >= y[i] - p(X[i, :])
            [i in 1:npoints], z[i] >= -y[i] + p(X[i, :])
        end)
    end

    return (model, p)
end

function run_JuMP_shapeconregr(use_wsos::Bool)
    (n, deg, npoints, signal_ratio, f) =
        (2, 3, 100, 0.0, x -> sum(x.^3)) # no noise, monotonic function
        # (2, 3, 100, 0.0, x -> sum(x.^4)) # no noise, non-monotonic function
        # (2, 3, 100, 50.0, x -> sum(x.^3)) # some noise, monotonic function
        # (2, 3, 100, 50.0, x -> sum(x.^4)) # some noise, non-monotonic function

    mono_dom = Hypatia.Box(-ones(n), ones(n))
    conv_dom = Hypatia.Box(-ones(n), ones(n))
    mono_profile = ones(n)
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)

    if use_wsos
        (model, p) = build_shapeconregr_WSOS(X, y, deg, mono_dom, conv_dom, mono_profile)
    else
        (model, p) = build_shapeconregr_PSD(X, y, deg, mono_dom, conv_dom, mono_profile)
    end

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

    return (pobj, p)
end

run_JuMP_shapeconregr_PSD() = run_JuMP_shapeconregr(false)
run_JuMP_shapeconregr_WSOS() = run_JuMP_shapeconregr(true)

p = run_JuMP_shapeconregr_WSOS()
