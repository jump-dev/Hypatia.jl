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
import Distributions
using MathOptInterface
MOI = MathOptInterface
using JuMP
using Hypatia
using MultivariatePolynomials
using DynamicPolynomials
using SemialgebraicSets
using SumOfSquares
using PolyJuMP
using MathOptInterfaceMosek
using Test


# a description of the shape of the regressor
mutable struct ShapeData
    mono_dom::Hypatia.InterpDomain
    conv_dom::Hypatia.InterpDomain
    mono_profile::Vector{Int}
    conv_profile::Int
end

ShapeData(n::Int) = ShapeData(
    Hypatia.Box(-ones(n), ones(n)),
    Hypatia.Box(-ones(n), ones(n)),
    ones(Int, n),
    1,
    )

# problem data
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
    X = rand(Distributions.Uniform(xmin, xmax), npoints, n)
    y = [func(X[p,:]) for p in 1:npoints]

    if !iszero(signal_ratio)
        noise = rand(Distributions.Normal(), npoints)
        noise .*= norm(y)/sqrt(signal_ratio)/norm(noise)
        y .+= noise
    end

    return (X, y)
end

function build_shapeconregr_PSD(
    X,
    y,
    r::Int,
    sd::ShapeData;
    use_leastsqobj::Bool = false,
    ignore_mono::Bool = false,
    ignore_conv::Bool = false,
    use_hypatia::Bool = true,
    )
    (mono_dom, conv_dom, mono_profile, conv_profile) = (sd.mono_dom, sd.conv_dom, sd.mono_profile, sd.conv_profile)
    (npoints, n) = size(X)

    @polyvar x[1:n]

    if use_hypatia
        model = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
    else
        model = SOSModel(with_optimizer(MosekOptimizer))
    end
    @variable(model, p, PolyJuMP.Poly(monomials(x, 0:r)))
    dp = [DynamicPolynomials.differentiate(p, x[i]) for i in 1:n]

    if !ignore_mono
        mono_bss = Hypatia.Hypatia.get_bss(mono_dom, x)
        for j in 1:n
            if !iszero(sd.mono_profile[j])
                @constraint(model, mono_profile[j] * dp[j] >= 0, domain=mono_bss)
            end
        end
    end

    # TODO think about what it means if wsos polynomials have degree > 2
    if !ignore_conv
        conv_bss = Hypatia.Hypatia.get_bss(conv_dom, x)
        Hp = [DynamicPolynomials.differentiate(dp[i], x[j]) for i in 1:n, j in 1:n]
        @SDconstraint(model, conv_profile * Hp >= 0, domain=conv_bss)
    end

    if use_leastsqobj
        @variable(model, z)
        @objective(model, Min, z / npoints)
        @constraint(model, [z, [y[i] - p(X[i, :]) for i in 1:npoints]...] in MOI.SecondOrderCone(1+npoints))
     else
        @variable(model, z[1:npoints])
        @objective(model, Min, sum(z) / npoints)
        @constraints(model, begin
            [i in 1:npoints], z[i] >= y[i] - p(X[i, :])
            [i in 1:npoints], z[i] >= -y[i] + p(X[i, :])
        end)
    end

    return (model, p)
end

function build_shapeconregr_WSOS(
    X,
    y,
    r::Int,
    sd::ShapeData;
    use_leastsqobj::Bool = false,
    ignore_mono::Bool = false,
    ignore_conv::Bool = false,
    mono_maxd::Int = -1,
    conv_maxd::Int = -1,
    sample_pts::Bool = true,
    )
    println("in jump model")
    (mono_dom, conv_dom, mono_profile, conv_profile) = (sd.mono_dom, sd.conv_dom, sd.mono_profile, sd.conv_profile)
    d = div(r, 2)
    (npoints, n) = size(X)

    @polyvar x[1:n]
    @polyvar w[1:n]

    model = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @elapsed @variable(model, p, PolyJuMP.Poly(monomials(x, 0:r)))

    if use_leastsqobj
        println("least squares objective")
        @variable(model, z)
        @objective(model, Min, z / npoints)
        @elapsed  @constraint(model, [z, [y[i] - p(X[i, :]) for i in 1:npoints]...] in MOI.SecondOrderCone(1+npoints))
     else
        @variable(model, z[1:npoints])
        @objective(model, Min, sum(z) / npoints)
        @constraints(model, begin
            [i in 1:npoints], z[i] >= y[i] - p(X[i, :])
            [i in 1:npoints], z[i] >= -y[i] + p(X[i, :])
        end)
    end

    # monotonicity
    dp = [DynamicPolynomials.differentiate(p, x[j]) for j in 1:n]
    if !ignore_mono
        println("monotonicity constraint")
        if sample_pts
            (mono_U, mono_pts, mono_P0, mono_PWts, _) = Hypatia.interp_sample(mono_dom, n, d, pts_factor=50)
        else
            (mono_U, mono_pts, mono_P0, _) = Hypatia.interp_box(n, d)
            P0sub = view(mono_P0, :, 1:binomial(n+d-1, n))
            (ubs, lbs) = (mono_dom.u, mono_dom.l)
            pscale = 0.5*(ubs - lbs)
            Wtsfun = (j -> sqrt.(1.0 .- abs2.(mono_pts[:,j]))*pscale[j])
            mono_PWts = [Wtsfun(j) .* P0sub for j in 1:n]
            mono_pts = mono_pts .* pscale' .+ 0.5*(ubs + lbs)'
        end
        mono_wsos_cone = WSOSPolyInterpCone(mono_U, [mono_P0, mono_PWts...])
        @elapsed for j in 1:n
            if abs(mono_profile[j]) > 0.5
                @constraint(model, [mono_profile[j] * dp[j](mono_pts[i, :]) for i in 1:mono_U] in mono_wsos_cone)
            end
        end
    end

    # convexity
    if !ignore_conv
        println("convexity constraint")
        full_conv_dom = Hypatia.addfreevars(conv_dom)
        if sample_pts
            (conv_U, conv_pts, conv_P0, conv_PWts, _) = Hypatia.interp_sample(full_conv_dom, 2n, d+1, pts_factor=50) # TODO think about if it's ok to go up to d+1
        else
            (conv_U, conv_pts, conv_P0, _) = Hypatia.interp_box(2n, d+1)
            P0sub = view(conv_P0, :, 1:binomial(n+d-1, n))
            (ubs, lbs) = (repeat(conv_dom.u, 2), repeat(conv_dom.l, 2))
            pscale = 0.5*(ubs - lbs)
            Wtsfun = (j -> sqrt.(1.0 .- abs2.(conv_pts[:,j]))*pscale[j])
            conv_PWts = [Wtsfun(j) .* P0sub for j in 1:n]
            conv_pts = conv_pts .* pscale' .+ 0.5*(ubs + lbs)'
        end

        conv_wsos_cone = WSOSPolyInterpCone(conv_U, [conv_P0, conv_PWts...])
        Hp = [DynamicPolynomials.differentiate(dp[i], x[j]) for i in 1:n, j in 1:n]
        conv_condition = w'*Hp*w
        @constraint(model, [conv_profile * conv_condition(conv_pts[i, :]) for i in 1:conv_U] in conv_wsos_cone)
    end

    return (model, p)
end

function run_JuMP_shapeconregr(use_wsos::Bool; usedense::Bool=true)
    (n, deg, npoints, signal_ratio, f) =
        # (2, 3, 100, 0.0, x -> exp(norm(x))) # no noise, monotonic function
        (2, 3, 100, 0.0, x -> sum(x.^3)) # no noise, monotonic function
        # (2, 3, 100, 0.0, x -> sum(x.^4)) # no noise, non-monotonic function
        # (2, 3, 100, 50.0, x -> sum(x.^3)) # some noise, monotonic function
        # (2, 3, 100, 50.0, x -> sum(x.^4)) # some noise, non-monotonic function
        # (2, 8, 100, 0.0, x -> exp(norm(x))) # low n high deg, numerically harder
        # (5, 5, 100, 0.0, x -> exp(norm(x))) # moderate size, no noise, monotonic # out of memory with psd

    shapedata = ShapeData(n)
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)

    use_leastsqobj = true

    if use_wsos
        (model, p) = build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=use_leastsqobj, usedense=usedense)
    else
        (model, p) = build_shapeconregr_PSD(X, y, deg, shapedata, use_leastsqobj=use_leastsqobj, usedense=usedense)
    end

    println("starting to solve JuMP model")
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
