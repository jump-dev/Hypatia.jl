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
    mono_profile::Vector{Float64},
    )
    (npoints, n) = size(X)
    @polyvar x[1:n]
    mono_bss = Hypatia.Hypatia.get_bss(mono_dom, x)
    conv_bss = Hypatia.Hypatia.get_bss(conv_dom, x)

    model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variables(model, begin
        p, PolyJuMP.Poly(monomials(x, 0:r))
        z[1:npoints]
    end)
    @objective(model, Min, sum(z))
    @constraints(model, begin
        [i in 1:npoints], z[i] >= y[i] - p(X[i, :])
        [i in 1:npoints], z[i] >= -y[i] + p(X[i, :])
    end)

    dp = [DynamicPolynomials.differentiate(p, x[i]) for i in 1:n]
    @constraint(model, [j in 1:n], mono_profile[j] * dp[j] >= 0, domain=mono_bss)

    # TODO think about what it means if wsos polynomials have degree > 2
    Hp = [DynamicPolynomials.differentiate(dp[i], x[j]) for i in 1:n, j in 1:n]
    @SDconstraint(model, Hp >= 0, domain=conv_bss)

    return (model, p)
end

function getregrinterp(
    d::Int,
    npoints::Int,
    n::Int,
    dom::Hypatia.InterpDomain,
    replicate_dims::Bool,
    pts_factor::Int,
    )
    L = binomial(n+d,n)
    U = binomial(n+2d, n)
    candidate_pts = Hypatia.interp_sample(dom, U * pts_factor)

    # TODO temporary hack, replace this with methods for unions of domains
    if replicate_dims
        candidate_pts2 = Hypatia.interp_sample(dom, U * pts_factor)
        candidate_pts = hcat(candidate_pts, candidate_pts2)
    end

    (M, _) = Hypatia.get_large_P(candidate_pts, d, U)
    F = qr!(Array(M'), Val(true))
    keep_pnt = F.p[1:U]
    pts = candidate_pts[keep_pnt,:] # subset of points indexed with the support of w
    P0 = M[keep_pnt, 1:L] # subset of polynomial evaluations up to total degree d
    P = Array(qr(P0).Q)
    P0sub = view(P0, :, 1:binomial(n+d-1, n))

    return (U, pts, P, P0sub)
end

function build_shapeconregr_WSOS(
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    mono_dom::Hypatia.InterpDomain,
    conv_dom::Hypatia.InterpDomain,
    mono_profile::Vector{Float64};
    ortho_wts::Bool = false,
    )
    @assert mod(r, 2) == 1
    d = div(r-1, 2)
    (npoints, n) = size(X)
    @polyvar x[1:n]
    @polyvar w[1:n]

    (mono_U, mono_pts, mono_P, mono_P0sub) = getregrinterp(d, npoints, n, mono_dom, false, 10n)
    mono_bss = Hypatia.get_bss(mono_dom, x)
    mono_g = Hypatia.get_weights(mono_dom, mono_pts)
    @assert length(mono_g) == length(mono_bss.p)
    mono_PWts = [sqrt.(gi) .* mono_P0sub for gi in mono_g]
    # if ortho_wts
    #     mono_PWts = [Array(qr!(W).Q) for W in mono_PWts] # orthonormalize
    # end
    mono_wsos_cone = WSOSPolyInterpCone(mono_U, [mono_P, mono_PWts...])

    # TODO think about if it's ok to go up to d+1
    (conv_U, conv_pts, conv_P, conv_P0sub) = getregrinterp(d+1, npoints, 2n, conv_dom, true, 10n)
    conv_bss = Hypatia.get_bss(conv_dom, x)
    conv_g = Hypatia.get_weights(conv_dom, conv_pts; count=n)
    @assert length(conv_g) == length(conv_bss.p)
    conv_PWts = [sqrt.(gi) .* conv_P0sub for gi in conv_g]
    # if ortho_wts
    #     conv_PWts = [Array(qr!(W).Q) for W in conv_PWts] # orthonormalize
    # end
    conv_wsos_cone = WSOSPolyInterpCone(conv_U, [conv_P, conv_PWts...])

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @variables(model, begin
        p, PolyJuMP.Poly(monomials(x, 0:r)) # monomial basis coefficients
        z[1:npoints] # residuals
        # interpolant basis coefficients
        # mono_interp_coeffs[1:mono_U, 1:n]
        conv_interp_coeffs[1:conv_U] # TODO symmetry
    end)
    @objective(model, Min, sum(z))
    @constraints(model, begin
        [i in 1:npoints], z[i] >= y[i] - p(X[i, :])
        [i in 1:npoints], z[i] >= -y[i] + p(X[i, :])
    end)

    # relate coefficients for monotonicity
    dp = [DynamicPolynomials.differentiate(p, x[j]) for j in 1:n]
    # for j in 1:n, i in 1:mono_U
    #     @constraint(model, mono_profile[j] * dp[j](mono_pts[i, :]) == mono_interp_coeffs[i, j])
    # end

    # relate coefficients for convexity
    Hp = [DynamicPolynomials.differentiate(dp[i], x[j]) for i in 1:n, j in 1:n]
    conv_condition = w'*Hp*w
    @constraints(model, begin
        [j in 1:n], [mono_profile[j] * dp[j](mono_pts[i, :]) for i in 1:mono_U] in mono_wsos_cone
        # [conv_condition(conv_pts[i, :]) for i in 1:conv_U] in conv_wsos_cone
        [i in 1:conv_U], conv_condition(conv_pts[i, :]) == conv_interp_coeffs[i]
        # [j in 1:n], mono_interp_coeffs[:, j] in mono_wsos_cone
        conv_interp_coeffs in conv_wsos_cone
    end)

    return (model, p)
end

function run_JuMP_shapeconregr(use_wsos::Bool)
    (n, deg, npoints, signal_ratio, f) =
        (2, 3, 100, 0.0, x -> sum(x.^3)) # no noise, monotonic function
        # (2, 3, 100, 0.0, x -> sum(x.^4)) # no noise, non-monotonic function
        # (2, 3, 100, 50.0, x -> sum(x.^3)) # some noise, monotonic function
        # (2, 3, 100, 50.0, x -> sum(x.^4)) # some noise, non-monotonic function

    conv_dom = mono_dom = Hypatia.Box(-ones(n), ones(n))
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

    preds = [JuMP.value(p)(X[i,:]) for i in 1:npoints]

    return (pobj, preds)
end

run_JuMP_shapeconregr_PSD() = run_JuMP_shapeconregr(false)
run_JuMP_shapeconregr_WSOS() = run_JuMP_shapeconregr(true)
