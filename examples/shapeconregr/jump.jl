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
const PJ = PolyJuMP
import Random
import Distributions
import LinearAlgebra: norm
using Test

const rt2 = sqrt(2)

include("shapeconregrdata.jl")


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

    JuMP.@variable(model, p, PJ.Poly(DP.monomials(x, 0:deg)))
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

    monotonic_set = MU.get_domain_inequalities(shape_data.mono_dom, x)
    convex_set = MU.get_domain_inequalities(shape_data.conv_dom, x)

    # monotonicity
    for j in 1:n
        if !iszero(shape_data.mono_profile[j])
            gradient = DP.differentiate(p, x[j])
            JuMP.@constraint(model, shape_data.mono_profile[j] * gradient >= 0, domain = monotonic_set)
        end
    end

    # convexity
    if !iszero(shape_data.conv_profile)
        hessian = DP.differentiate(p, x, 2)
        JuMP.@constraint(model, shape_data.conv_profile * hessian in JuMP.PSDCone(), domain = convex_set)
    end

    return model
end

function build_shapeconregr_WSOS_PolyJuMP(
    model,
    X::Matrix{Float64},
    y::Vector{Float64},
    r::Int,
    shape_data::ShapeData;
    use_lsq_obj::Bool = true,
    sample::Bool = true,
    rseed::Int = 1,
    )
    Random.seed!(rseed)
    d = div(r + 1, 2)
    n = size(X, 2)

    (mono_U, mono_pts, mono_P0, mono_PWts, _) = MU.interpolate(shape_data.mono_dom, d, sample = sample, sample_factor = 50)
    (conv_U, conv_pts, conv_P0, conv_PWts, _) = MU.interpolate(shape_data.conv_dom, d - 1, sample = sample, sample_factor = 50)
    mono_wsos_cone = HYP.WSOSPolyInterpCone(mono_U, [mono_P0, mono_PWts...])
    conv_wsos_cone = HYP.WSOSPolyInterpMatCone(n, conv_U, [conv_P0, conv_PWts...])

    (x, p) = add_loss_and_polys(model, X, y, r, use_lsq_obj)

    # monotonicity
    for j in 1:n
        if !iszero(shape_data.mono_profile[j])
            gradient = DynamicPolynomials.differentiate(p, x[j])
            JuMP.@constraint(model, [shape_data.mono_profile[j] * gradient(mono_pts[u, :]) for u in 1:mono_U] in mono_wsos_cone)
        end
    end

    # convexity
    if !iszero(shape_data.conv_profile)
        hessian = DynamicPolynomials.differentiate(p, x, 2)
        JuMP.@constraint(model, [shape_data.conv_profile * hessian[i, j](conv_pts[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:conv_U] in conv_wsos_cone)
    end

    return model
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

    gradient_d = div(regressor_deg, 2)
    hessian_d = div(regressor_deg - 1, 2)
    (num_points, n) = size(X)

    (regressor_points, _) = MU.get_interp_pts(MU.FreeDomain(n), regressor_deg, sample_factor = 50)
    regressor_U = size(regressor_points, 1)

    (mono_U, mono_points, mono_P0, mono_PWts, _) = MU.interpolate(shape_data.mono_dom, gradient_d, sample = sample, sample_factor = 50)
    (conv_U, conv_points, conv_P0, conv_PWts, _) = MU.interpolate(shape_data.conv_dom, hessian_d, sample = sample, sample_factor = 50)
    mono_wsos_cone = HYP.WSOSPolyInterpCone(mono_U, [mono_P0, mono_PWts...])
    conv_wsos_cone = HYP.WSOSPolyInterpMatCone(n, conv_U, [conv_P0, conv_PWts...])

    lagrange_polys = MU.recover_lagrange_polys(regressor_points, regressor_deg)

    JuMP.@variable(model, regressor, variable_type = PJ.Poly(PJ.FixedPolynomialBasis(lagrange_polys)))
    if use_lsq_obj
        JuMP.@variable(model, z)
        JuMP.@objective(model, Min, z / num_points)
        JuMP.@constraint(model, vcat([z], [y[i] - regressor(X[i, :]) for i in 1:num_points]) in MOI.SecondOrderCone(1 + num_points))
     else
        JuMP.@variable(model, z[1:num_points])
        JuMP.@objective(model, Min, sum(z) / num_points)
        JuMP.@constraints(model, begin
            [i in 1:num_points], z[i] >= y[i] - regressor(X[i, :])
            [i in 1:num_points], z[i] >= -y[i] + regressor(X[i, :])
        end)
    end

    # monotonicity
    for j in 1:n
        if !iszero(shape_data.mono_profile[j])
            gradient = DP.differentiate(regressor, DP.variables(regressor)[j])
            JuMP.@constraint(model, [shape_data.mono_profile[j] * gradient(mono_points[u, :]) for u in 1:mono_U] in mono_wsos_cone)
        end
    end

    # convexity
    if !iszero(shape_data.conv_profile)
        hessian = DP.differentiate(regressor, DP.variables(regressor), 2)
        JuMP.@constraint(model, [shape_data.conv_profile * hessian[i, j](conv_points[u, :]) * (i == j ? 1.0 : rt2)
            for i in 1:n for j in 1:i for u in 1:conv_U] in conv_wsos_cone)
    end

    return model
end

function shapeconregr_JuMP(inst::Int; use_dense::Bool = true, use_PolyJuMP::Bool = false, use_wsos::Bool = true)
    (n, deg, num_points, signal_ratio, f, shapedata, use_lsq_ob_j, true_obj) = getshapeconregrdata(inst)
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true, use_dense = use_dense))
    if use_wsos
        if use_PolyJuMP
            model = build_shapeconregr_WSOS_PolyJuMP(model, X, y, deg, shapedata, use_lsq_obj = use_lsq_obj)
        else
            model = build_shapeconregr_WSOS(model, X, y, deg, shapedata, use_lsq_obj = use_lsq_obj)
        end
    else
        PJ.setpolymodule!(model, SumOfSquares)
        model = build_shapeconregr_PSD(model, X, y, deg, shapedata, use_lsq_obj = use_lsq_obj)
    end
    return (model, true_obj)
end

shapeconregr1_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(1, use_dense = use_dense)
shapeconregr2_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(2, use_dense = use_dense)
shapeconregr3_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(3, use_dense = use_dense)
shapeconregr4_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(4, use_dense = use_dense)
shapeconregr5_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(5, use_dense = use_dense)
shapeconregr6_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(6, use_dense = use_dense)
shapeconregr7_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(7, use_dense = use_dense)
shapeconregr8_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(8, use_dense = use_dense)
shapeconregr9_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(9, use_dense = use_dense)
shapeconregr10_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(10, use_dense = use_dense)
shapeconregr11_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(11, use_dense = use_dense)
shapeconregr12_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(12, use_dense = use_dense, use_wsos = false)
shapeconregr13_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(13, use_dense = use_dense)
shapeconregr14_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(13, use_dense = use_dense, use_wsos = false)
shapeconregr15_JuMP(; use_dense::Bool = true) = shapeconregr_JuMP(15, use_dense = use_dense, use_PolyJuMP = true)

function test_shapeconregr_JuMP(instance::Function)
    (model, true_obj) = instance()

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
    @test primal_obj ≈ true_obj atol = 1e-4 rtol = 1e-4

    return
end

test_shapeconregr_JuMP_all() = test_shapeconregr_JuMP.([
    shapeconregr1_JuMP,
    shapeconregr2_JuMP,
    shapeconregr3_JuMP,
    shapeconregr4_JuMP,
    shapeconregr5_JuMP,
    shapeconregr6_JuMP,
    shapeconregr7_JuMP,
    shapeconregr8_JuMP,
    shapeconregr9_JuMP,
    shapeconregr10_JuMP,
    shapeconregr11_JuMP,
    shapeconregr12_JuMP,
    shapeconregr13_JuMP,
    # shapeconregr14_JuMP,
    shapeconregr15_JuMP,
    ])

test_shapeconregr_JuMP_small() = test_shapeconregr_JuMP.([
    shapeconregr1_JuMP,
    shapeconregr2_JuMP,
    shapeconregr12_JuMP,
    shapeconregr15_JuMP,
    ])
