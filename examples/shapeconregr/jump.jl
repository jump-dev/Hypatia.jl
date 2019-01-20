#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

given data (xᵢ, yᵢ), find a polynomial p to solve
    min ∑ᵢℓ(p(xᵢ), yᵢ)
    ρⱼ × dᵏp/dtⱼᵏ ≥ 0 ∀ t ∈ D
where
    - dᵏp/dtⱼᵏ is the kᵗʰ derivative of p in direction j,
    - ρⱼ determines the desired sign of the derivative,
    - D is a domain such as a box or an ellipsoid,
    - ℓ is a convex loss function.
see e.g. Chapter 8 of thesis by G. Hall (2018)
=#

import Hypatia
const HYP = Hypatia
const CO = HYP.Cones
const LS = HYP.LinearSystems
const MU = HYP.ModelUtilities

import MathOptInterface
const MOI = MathOptInterface
import JuMP
import MultivariatePolynomials
import DynamicPolynomials
import SumOfSquares
import PolyJuMP
import Random
import Distributions
using Test

const rt2 = sqrt(2)

# a description of the shape of the regressor
mutable struct ShapeData
    mono_dom::MU.Domain
    conv_dom::MU.Domain
    mono_profile::Vector{Int}
    conv_profile::Int
end
ShapeData(n::Int) = ShapeData(
    MU.Box(-ones(n), ones(n)), MU.Box(-ones(n), ones(n)),
    ones(Int, n), 1)

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
    y = [func(X[p, :]) for p in 1:npoints]
    if !iszero(signal_ratio)
        noise = rand(Distributions.Normal(), npoints)
        noise .*= norm(y) / sqrt(signal_ratio) / norm(noise)
        y .+= noise
    end
    return (X, y)
end

function add_loss_and_polys!(
    model::JuMP.Model,
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    use_leastsqobj::Bool
    )
    (npoints, n) = size(X)
    DynamicPolynomials.@polyvar x[1:n]
    JuMP.@variable(model, p, PolyJuMP.Poly(DynamicPolynomials.monomials(x, 0:r)))
    if use_leastsqobj
        JuMP.@variable(model, z)
        JuMP.@objective(model, Min, z / npoints)
        JuMP.@constraint(model, [z, [y[i] - p(X[i, :]) for i in 1:npoints]...] in MOI.SecondOrderCone(1+npoints))
     else
        JuMP.@variable(model, z[1:npoints])
        JuMP.@objective(model, Min, sum(z) / npoints)
        JuMP.@constraints(model, begin
            [i in 1:npoints], z[i] >= y[i] - p(X[i, :])
            [i in 1:npoints], z[i] >= -y[i] + p(X[i, :])
        end)
    end
    return (x, p)
end

function build_shapeconregr_PSD(
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    sd::ShapeData;
    use_leastsqobj::Bool = false,
    usedense::Bool = true,
    )
    n = size(X, 2)
    d = div(r + 1, 2)

    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose=true, usedense=usedense, linearsystem=LS.QRSymm))
    (x, p) = add_loss_and_polys!(model, X, y, r, use_leastsqobj)

    mono_bss = MU.get_domain_inequalities(sd.mono_dom, x)
    conv_bss = MU.get_domain_inequalities(sd.conv_dom, x)

    # monotonicity
    for j in 1:n
        if !iszero(sd.mono_profile[j])
            dpj = DynamicPolynomials.differentiate(p, x[j])
            JuMP.@constraint(model, sd.mono_profile[j] * dpj >= 0, domain=mono_bss)
        end
    end

    # convexity
    if !iszero(sd.conv_profile)
        # TODO think about what it means if wsos polynomials have degree > 2
        Hp = DynamicPolynomials.differentiate(p, x, 2)
        JuMP.@SDconstraint(model, sd.conv_profile * Hp >= 0, domain=conv_bss)
    end

    return (model, p)
end

function build_shapeconregr_WSOS(
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    sd::ShapeData;
    use_leastsqobj::Bool = false,
    usedense::Bool = true,
    sample::Bool = true,
    )
    d = div(r + 1, 2)
    n = size(X, 2)

    (mono_U, mono_pts, mono_P0, mono_PWts, _) = MU.interpolate(sd.mono_dom, d, sample=sample, sample_factor=50)
    (conv_U, conv_pts, conv_P0, conv_PWts, _) = MU.interpolate(sd.conv_dom, d - 1, sample=sample, sample_factor=50)
    mono_wsos_cone = HYP.WSOSPolyInterpCone(mono_U, [mono_P0, mono_PWts...])
    conv_wsos_cone = HYP.WSOSPolyInterpMatCone(n, conv_U, [conv_P0, conv_PWts...])

    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose=true, usedense=usedense, linearsystem=LS.QRSymm, tolabsopt=1e-6, tolrelopt=1e-5, tolfeas=1e-6))
    (x, p) = add_loss_and_polys!(model, X, y, r, use_leastsqobj)

    # monotonicity
    for j in 1:n
        if !iszero(sd.mono_profile[j])
            dpj = DynamicPolynomials.differentiate(p, x[j])
            JuMP.@constraint(model, [sd.mono_profile[j] * dpj(mono_pts[u, :]) for u in 1:mono_U] in mono_wsos_cone)
        end
    end

    # convexity
    if !iszero(sd.conv_profile)
        Hp = DynamicPolynomials.differentiate(p, x, 2)
        JuMP.@constraint(model, [sd.conv_profile * Hp[i,j](conv_pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:conv_U] in conv_wsos_cone)
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
        # (2, 4, 100, 0.0, x -> -sum(x.^4))
        # (2, 4, 100, 0.0, x -> sum(x)^2)

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

    @test term_status == MOI.OPTIMAL
    @test pr_status == MOI.FEASIBLE_POINT
    @test du_status == MOI.FEASIBLE_POINT
    @test pobj ≈ dobj atol=1e-4 rtol=1e-4

    return (pobj, p)
end

run_JuMP_shapeconregr_PSD() = run_JuMP_shapeconregr(false)
run_JuMP_shapeconregr_WSOS() = run_JuMP_shapeconregr(true)
