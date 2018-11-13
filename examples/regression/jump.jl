#=
Copyright 2018, Chris Coey and contributors

Given data (x_i, y_i) find a polynomial p such that
rho * dp/dx_j ≥ 0 ∀ x ∈ B
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
using Test

function regression_data(
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

function build_regression_SDP(
    X,
    y,
    r::Int,
    box_lower::Vector{Float64},
    box_upper::Vector{Float64},
    rho::Vector{Float64} = zeros(size(X, 2)),
    )

    (npoints, n) = size(X)

    mdl = SOSModel(with_optimizer(Hypatia.Optimizer, verbose=true))
    @polyvar x[1:n]
    PX = monomials(x, 0:r)

    @variable mdl p PolyJuMP.Poly(PX)
    dp = [differentiate(p, x[i]) for i in 1:n]
    @variable mdl z[1:npoints]

    dom = BasicSemialgebraicSet{Float64,Polynomial{true,Float64}}()
    for xi in x
        addinequality!(dom, -xi + 1.0)
        addinequality!(dom, xi + 1.0)
    end
    for j in 1:n
        if abs(rho[j]) > 0.5
            @constraint(mdl, rho[j] * dp[j] >= 0, domain=dom)
        end
    end
    @constraint mdl [i=1:npoints] z[i] >= y[i] - p(X[i, :])
    @constraint mdl [i=1:npoints] z[i] >= -y[i] + p(X[i, :])

    @objective mdl Min sum(z)

    JuMP.optimize!(mdl)

    return mdl, p
end

function build_regression_WSOS(
    X,
    y,
    r::Int,
    box_lower::Vector{Float64},
    box_upper::Vector{Float64},
    rho::Vector{Float64} = zeros(size(X, 2)),
    )

    @assert mod(r, 2) == 1

    (npoints, n) = size(X)
    d = ceil(Int, (r - 1) * 0.5)
    (L, U, pts, P0, P, w) = Hypatia.interpolate(n, d, calc_w=true)
    P0sub = view(P0, :, 1:binomial(n+d-1, n))
    pscale = 0.5*(box_upper - box_lower)
    Wtsfun = (j -> sqrt.(1.0 .- abs2.(pts[:,j]))*pscale[j])
    PWts = [Wtsfun(j) .* P0sub for j in 1:n]

    mdl = Model(with_optimizer(Hypatia.Optimizer, verbose=true))
    @polyvar x[1:n]
    PX = monomials(x, 0:r)

    # monomial basis coefficients
    @variable mdl p PolyJuMP.Poly(PX)
    dp = [differentiate(p, x[j]) for j in 1:n]
    @variable mdl z[1:npoints]

    # interpolant basis coefficients
    @variable mdl q[1:U, 1:n]

    # Vandermonde matrix to relate coefficients
    for j in 1:n
        for i in 1:U
            if abs(rho[j]) > 0.5
                @constraint mdl rho[j] * dp[j](pts[i, :]) == q[i, j]
            end
        end
    end

    @constraint mdl [j=1:n] q[:, j] in WSOSPolyInterpCone(U, [P, PWts...])

    # minimize sum of absolute error
    @constraint mdl [i=1:npoints] z[i] >= y[i] - p(X[i, :])
    @constraint mdl [i=1:npoints] z[i] >= -y[i] + p(X[i, :])

    @objective mdl Min sum(z)

    JuMP.optimize!(mdl)

    return mdl, p
end

function run_JuMP_regression()
    # degree of regressor
    r = 5
    # dimensionality of observations
    n = 2
    npoints = binomial(n + r, n) * 10
    box_lower = fill(-1.0, n)
    box_upper = fill(1.0, n)
    # monotonicity everywhere
    monotonicity_profile = ones(n)
    f = (x -> sum(x.^4, dims=2))
    (X, y) = regression_data(f, npoints=npoints, signal_ratio=0.0, n=n)

    sdp_mdl, sdp_p = build_regression_SDP(X, y, r, box_lower, box_upper, monotonicity_profile)
    wsos_mdl, wsos_p = build_regression_WSOS(X, y, r, box_lower, box_upper, monotonicity_profile)
    @test JuMP.objective_value(sdp_mdl) ≈ JuMP.objective_value(wsos_mdl) atol = 1e-4

    sdp_preds = [JuMP.value(sdp_p)(X[i,:]) for i in 1:npoints]
    wsos_preds = [JuMP.value(wsos_p)(X[i,:]) for i in 1:npoints]
    @test sdp_preds ≈ wsos_preds atol = 1e-4

    (X, y) = regression_data(npoints=npoints, signal_ratio=50.0, n=n)

    sdp_mdl, sdp_p = build_regression_SDP(X, y, r, box_lower, box_upper, monotonicity_profile)
    wsos_mdl, wsos_p = build_regression_WSOS(X, y, r, box_lower, box_upper, monotonicity_profile)

    @test JuMP.objective_value(sdp_mdl) ≈ JuMP.objective_value(wsos_mdl) atol = 1e-5

    sdp_preds = [JuMP.value(sdp_p)(X[i,:]) for i in 1:npoints]
    wsos_preds = [JuMP.value(wsos_p)(X[i,:]) for i in 1:npoints]
    @test sdp_preds ≈ wsos_preds atol = 1e-4

    return nothing
end
