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
using JuMP
using PolyJuMP
using MultivariatePolynomials
using SemialgebraicSets
using SumOfSquares
using DynamicPolynomials
using Hypatia
using PlotlyJS
using ORCA
using Test

include(joinpath(dirname(@__DIR__()), "domains.jl"))

function shapeconregr_data(
    func::Function = (x -> sum(x.^3, dims=2));
    rseed::Int = 1,
    xmin::Float64 = -1.0,
    xmax::Float64 = 1.0,
    n::Int = 1,
    npoints::Int = 1,
    signal_ratio::Float64 = 1.0,
    )

    Random.seed!(1234)

    X = rand(Uniform(xmin, xmax), npoints, n)
    y = func(X)
    if signal_ratio > 1e-3
        noise = rand(Normal(), npoints)
        noise .*= (norm(y) / ( sqrt(signal_ratio) * norm(noise)))
        y .+= noise
    end

    return (X, y)
end

function build_shapeconregr_SDP(
    X,
    y,
    r::Int,
    dom::Domain,
    rho::Vector{Float64} = zeros(size(X, 2)),
    )

    (npoints, n) = size(X)

    mdl = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
    @polyvar x[1:n]
    PX = monomials(x, 0:r)
    bss = get_bss(dom, x)

    @variable(mdl, p, PolyJuMP.Poly(PX))
    dp = [differentiate(p, x[i]) for i in 1:n]
    @variable(mdl, z[1:npoints])

    for j in 1:n
        if abs(rho[j]) > 0.5
            @constraint(mdl, rho[j] * dp[j] >= 0, domain=bss)
        end
    end
    @constraints(mdl, begin
        [i=1:npoints], z[i] >= y[i] - p(X[i, :])
        [i=1:npoints], z[i] >= -y[i] + p(X[i, :])
    end)

    @objective mdl Min sum(z)

    JuMP.optimize!(mdl)

    return mdl, p
end

function build_shapeconregr_WSOS(
    X,
    y,
    r::Int,
    dom::Domain,
    rho::Vector{Float64} = zeros(size(X, 2)),
    )

    @assert mod(r, 2) == 1
    d = div(r-1, 2)

    (npoints, n) = size(X)
    L = binomial(n+d,n)
    U = binomial(n+2d, n)
    pts_factor = n
    candidate_pts = sample(dom, U * pts_factor)
    M = get_P(candidate_pts, d, U)
    Mp = Array(M')
    F = qr!(Mp, Val(true))
    keep_pnt = F.p[1:U]
    pts = candidate_pts[keep_pnt,:] # subset of points indexed with the support of w
    P0 = M[keep_pnt,1:L] # subset of polynomial evaluations up to total degree d
    P = Array(qr(P0).Q)
    P0sub = view(P0, :, 1:binomial(n+d-1, n))

    mdl = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @polyvar x[1:n]
    PX = monomials(x, 0:r)

    bss = get_bss(dom, x)
    g = get_weights(dom, bss, pts)
    @assert length(g) == length(bss.p)
    PWts = [sqrt.(gi) .* P0sub for gi in g]

    wsos_cone = WSOSPolyInterpCone(U, [P, PWts...])

    # monomial basis coefficients
    @variable(mdl, p, PolyJuMP.Poly(PX))
    dp = [differentiate(p, x[j]) for j in 1:n]

    @variables(mdl, begin
        # residuals
        z[1:npoints]
        # interpolant basis coefficients
        q[1:U, 1:n]
    end)

    # Vandermonde matrix to relate coefficients
    for j in 1:n
        for i in 1:U
            if abs(rho[j]) > 0.5
                @constraint(mdl, rho[j] * dp[j](pts[i, :]) == q[i, j])
            end
        end
    end

    @constraints(mdl, begin
        [j=1:n], q[:, j] in WSOSPolyInterpCone(U, [P, PWts...])
        # minimize sum of absolute error
        [i=1:npoints], z[i] >= y[i] - p(X[i, :])
        [i=1:npoints], z[i] >= -y[i] + p(X[i, :])
    end)

    @objective mdl Min sum(z)

    JuMP.optimize!(mdl)

    return mdl, p
end

function run_JuMP_shapeconregr()
    # degree of regressor
    r = 5
    # dimensionality of observations
    n = 2
    npoints = binomial(n + r, n) * 10
    domain = Box(fill(-1.0, n), fill(1.0, n))
    # domain = Ball(zeros(n), 1.0)
    # monotonicity everywhere
    monotonicity_profile = ones(n)
    # f = (x -> sum(x .* (x .- 2), dims=2))
    f = (x -> sum(x.^2, dims=2))
    (X, y) = shapeconregr_data(f, npoints=npoints, signal_ratio=0.0, n=n)

    sdp_mdl, sdp_p = build_shapeconregr_SDP(X, y, r, domain, monotonicity_profile)
    wsos_mdl, wsos_p = build_shapeconregr_WSOS(X, y, r, domain, monotonicity_profile)
    # @test JuMP.objective_value(sdp_mdl) ≈ JuMP.objective_value(wsos_mdl) atol = 1e-4

    sdp_preds = [JuMP.value(sdp_p)(X[i,:]) for i in 1:npoints]
    wsos_preds = [JuMP.value(wsos_p)(X[i,:]) for i in 1:npoints]

    @test sdp_preds ≈ wsos_preds atol = 1e-4

    (X, y) = shapeconregr_data(npoints=npoints, signal_ratio=50.0, n=n)

    sdp_mdl, sdp_p = build_shapeconregr_SDP(X, y, r, domain, monotonicity_profile)
    wsos_mdl, wsos_p = build_shapeconregr_WSOS(X, y, r, domain, monotonicity_profile)

    @test JuMP.objective_value(sdp_mdl) ≈ JuMP.objective_value(wsos_mdl) atol = 1e-5

    sdp_preds = [JuMP.value(sdp_p)(X[i,:]) for i in 1:npoints]
    wsos_preds = [JuMP.value(wsos_p)(X[i,:]) for i in 1:npoints]
    @test sdp_preds ≈ wsos_preds atol = 1e-4

    return nothing
end

function makeplot()
    data_trace = scatter3d(
        x=X[:, 1],
        y=X[:, 2],
        z=y[:],
        mode="markers",
        opacity=0.8,
        marker_size=6,
        marker_line_width=0.5,
        marker_line_color="rgba(217, 217, 217, 0.14)"
    )
    randx = rand(Uniform(-1, 1), 200)
    randy = rand(Uniform(-1, 1), 200)
    sdpz = [JuMP.value(sdp_p)(hcat(randx, randy)[i,:]) for i in 1:200]
    wsosz = [JuMP.value(wsos_p)(hcat(randx, randy)[i,:]) for i in 1:200]
    sdp_trace = mesh3d(
        x=randx,
        y=randy,
        z=sdpz,
        mode="markers",
        opacity=0.4,
        marker_size=6,
        marker_line_width=0.5,
        marker_line_color="rgba(217, 217, 217, 0.14)"
    )
    wsos_trace = mesh3d(
        x=randx,
        y=randy,
        z=wsosz,
        mode="markers",
        opacity=0.4,
        marker_size=6,
        marker_line_width=0.5,
        marker_line_color="rgba(217, 217, 217, 0.14)"
    )
    layout = Layout(margin=attr(l=0, r=0, t=0, b=0))

    sdp_plot = plot([data_trace, sdp_trace], layout);
    wsos_plot = plot([data_trace, wsos_trace], layout);
    savefig(sdp_plot, "sdp_plot.pdf")
    savefig(wsos_plot, "wsos_plot.pdf")

end
