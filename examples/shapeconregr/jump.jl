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
const SO = HYP.Solvers
const MO = HYP.Models
const MU = HYP.ModelUtilities

import MathOptInterface
const MOI = MathOptInterface
import JuMP
import MultivariatePolynomials
import DynamicPolynomials
const DP = DynamicPolynomials
import SumOfSquares
import PolyJuMP
import Random
import Distributions
import LinearAlgebra: norm
using Test

const rt2 = sqrt(2)

# a description of the shape of the regressor
mutable struct ShapeData
    mono_dom::MU.Domain
    conv_dom::MU.Domain
    mono_profile::Vector{Int}
    conv_profile::Int
end
ShapeData(n::Int) = ShapeData(MU.Box(-ones(n), ones(n)), MU.Box(-ones(n), ones(n)), ones(Int, n), 1)

# problem data
function generate_regr_data(
    func::Function,
    xmin::Float64,
    xmax::Float64,
    n::Int,
    num_points::Int;
    signal_ratio::Float64 = 1.0,
    rseed::Int = 1,
    )
    Random.seed!(rseed)
    @assert 0.0 <= signal_ratio < Inf

    X = rand(Distributions.Uniform(xmin, xmax), num_points, n)
    y = [func(X[p, :]) for p in 1:num_points]
    if !iszero(signal_ratio)
        noise = rand(Distributions.Normal(), num_points)
        noise .*= norm(y) / sqrt(signal_ratio) / norm(noise)
        y .+= noise
    end

    return (X, y)
end

function add_loss_and_polys(
    model::JuMP.Model,
    X::Matrix{Float64},
    y::Vector{Float64},
    deg::Int,
    use_lsq_obj::Bool,
    )
    (num_points, n) = size(X)
    DP.@polyvar x[1:n]

    JuMP.@variable(model, p, PolyJuMP.Poly(DP.monomials(x, 0:deg)))
    if use_lsq_obj
        JuMP.@variable(model, z)
        JuMP.@objective(model, Min, z / num_points)
        JuMP.@constraint(model, vcat([z], [y[i] - p(X[i, :]) for i in 1:num_points]) in MOI.SecondOrderCone(1 + num_points))
     else
        JuMP.@variable(model, z[1:num_points])
        JuMP.@objective(model, Min, sum(z) / num_points)
        JuMP.@constraints(model, begin
            [i in 1:num_points], z[i] >= y[i] - p(X[i, :])
            [i in 1:num_points], z[i] >= -y[i] + p(X[i, :])
        end)
    end

    return (x, p)
end

function build_shapeconregr_PSD(
    model,
    X::Matrix{Float64},
    y::Vector{Float64},
    regressor_deg::Int,
    shape_data::ShapeData;
    use_lsq_obj::Bool = true,
    )
    n = size(X, 2)
    d = div(regressor_deg + 1, 2)

    (x, p) = add_loss_and_polys(model, X, y, regressor_deg, use_lsq_obj)

    mono_bss = MU.get_domain_inequalities(shape_data.mono_dom, x)
    conv_bss = MU.get_domain_inequalities(shape_data.conv_dom, x)

    # monotonicity
    for j in 1:n
        if !iszero(shape_data.mono_profile[j])
            gradient = DP.differentiate(p, x[j])
            JuMP.@constraint(model, shape_data.mono_profile[j] * gradient >= 0, domain = mono_bss)
        end
    end

    # convexity
    if !iszero(shape_data.conv_profile)
        hessian = DP.differentiate(p, x, 2)
        JuMP.@constraint(model, shape_data.conv_profile * hessian in JuMP.PSDCone(), domain = conv_bss)
    end

    return p
end

function build_shapeconregr_WSOS_PolyJuMP(
    model,
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    shapedata::ShapeData;
    use_lsq_obj::Bool = true,
    sample::Bool = true,
    rseed::Int = 1,
    )
    Random.seed!(rseed)
    d = div(r + 1, 2)
    n = size(X, 2)

    (mono_U, mono_pts, mono_P0, mono_PWts, _) = MU.interpolate(shapedata.mono_dom, d, sample = sample, sample_factor = 50)
    (conv_U, conv_pts, conv_P0, conv_PWts, _) = MU.interpolate(shapedata.conv_dom, d - 1, sample = sample, sample_factor = 50)
    mono_wsos_cone = HYP.WSOSPolyInterpCone(mono_U, [mono_P0, mono_PWts...])
    conv_wsos_cone = HYP.WSOSPolyInterpMatCone(n, conv_U, [conv_P0, conv_PWts...])

    (x, p) = add_loss_and_polys(model, X, y, r, use_lsq_obj)

    # monotonicity
    for j in 1:n
        if !iszero(shapedata.mono_profile[j])
            dpj = DynamicPolynomials.differentiate(p, x[j])
            JuMP.@constraint(model, [shapedata.mono_profile[j] * dpj(mono_pts[u, :]) for u in 1:mono_U] in mono_wsos_cone)
        end
    end

    # convexity
    if !iszero(shapedata.conv_profile)
        Hp = DynamicPolynomials.differentiate(p, x, 2)
        JuMP.@constraint(model, [shapedata.conv_profile * Hp[i, j](conv_pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:conv_U] in conv_wsos_cone)
    end

    return p
end

function build_shapeconregr_WSOS(
    model,
    X::Matrix{Float64},
    y::Vector{Float64},
    regressor_deg::Int,
    shape_data::ShapeData;
    use_lsq_obj::Bool = true,
    sample::Bool = true,
    rseed::Int = 1,
    )
    Random.seed!(rseed)

    derivative_d = div(regressor_deg, 2)
    hessian_d = div(regressor_deg - 1, 2)
    (num_points, n) = size(X)

    regressor_points = MU.get_interp_pts(MU.FreeDomain(n), regressor_deg, sample_factor = 50)
    regressor_U = size(regressor_points, 1)

    (mono_U, mono_points, mono_P0, mono_PWts, _) = MU.interpolate(shape_data.mono_dom, derivative_d, sample = sample, sample_factor = 50)
    (conv_U, conv_points, conv_P0, conv_PWts, _) = MU.interpolate(shape_data.conv_dom, hessian_d, sample = sample, sample_factor = 50)
    mono_wsos_cone = HYP.WSOSPolyInterpCone(mono_U, [mono_P0, mono_PWts...])
    conv_wsos_cone = HYP.WSOSPolyInterpMatCone(n, conv_U, [conv_P0, conv_PWts...])

    JuMP.@variable(model, regressor[1:regressor_U])
    lagrange_polys = MU.recover_lagrange_polys(regressor_points, regressor_deg)
    JuMP.@expression(model, regressor_poly, JuMP.dot(regressor, lagrange_polys))
    if use_lsq_obj
        JuMP.@variable(model, z)
        JuMP.@objective(model, Min, z / num_points)
        JuMP.@constraint(model, vcat([z], [y[i] - regressor_poly(X[i, :]) for i in 1:num_points]) in MOI.SecondOrderCone(1 + num_points))
     else
        JuMP.@variable(model, z[1:num_points])
        JuMP.@objective(model, Min, sum(z) / num_points)
        JuMP.@constraints(model, begin
            [i in 1:num_points], z[i] >= y[i] - regressor_poly(X[i, :])
            [i in 1:num_points], z[i] >= -y[i] + regressor_poly(X[i, :])
        end)
    end

    # monotonicity
    for j in 1:n
        if !iszero(shape_data.mono_profile[j])
            gradient = DP.differentiate(regressor_poly, DP.variables(regressor_poly)[j]) # TODO hack because lagrange polynomial may not have vars in all dimensions although unlikely
            JuMP.@constraint(model, [shape_data.mono_profile[j] * gradient(mono_points[u, :]) for u in 1:mono_U] in mono_wsos_cone)
        end
    end

    # convexity
    if !iszero(shape_data.conv_profile)
        hessian = DP.differentiate(regressor_poly, DP.variables(regressor_poly), 2)
        JuMP.@constraint(model, [shape_data.conv_profile * hessian[i, j](conv_points[u, :]) * (i == j ? 1.0 : rt2)
            for i in 1:n for j in 1:i for u in 1:conv_U] in conv_wsos_cone)
    end

    return regressor, lagrange_polys
end

function run_JuMP_shapeconregr(use_wsos::Bool; dense::Bool = true, use_PolyJuMP::Bool = false)
    (n, deg, num_points, signal_ratio, f) =
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
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)

    if use_wsos
        if use_PolyJuMP
            model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, use_dense = dense))
            p = build_shapeconregr_WSOS_PolyJuMP(model, X, y, deg, shapedata)
        else
            model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, use_dense = dense))
            (coeffs, polys) = build_shapeconregr_WSOS(model, X, y, deg, shapedata)
            p = JuMP.dot(coeffs, polys)
        end
    else
        model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true, use_dense = dense))
        p = build_shapeconregr_PSD(model, X, y, deg, shapedata)
    end

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

    return (primal_obj, p)
end

run_JuMP_shapeconregr_PSD() = run_JuMP_shapeconregr(false)
run_JuMP_shapeconregr_WSOS() = run_JuMP_shapeconregr(true)
run_JuMP_shapeconregr_WSOS_PolyJuMP() = run_JuMP_shapeconregr(true, use_PolyJuMP = true)
