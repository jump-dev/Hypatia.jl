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

import Random
import Distributions
import LinearAlgebra: norm
using Test
import DynamicPolynomials
const DP = DynamicPolynomials
import SumOfSquares
import PolyJuMP
const PJ = PolyJuMP
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia
const HYP = Hypatia
const MU = HYP.ModelUtilities

const rt2 = sqrt(2)

function generate_regr_data(
    n::Int,
    num_points::Int,
    f::Function;
    signal_ratio::Float64 = 0.0,
    xmin::Float64 = -1.0,
    xmax::Float64 = 1.0,
    rseed::Int = 1,
    )
    Random.seed!(rseed)
    X = rand(Distributions.Uniform(xmin, xmax), num_points, n)
    y = [f(X[p, :]) for p in 1:num_points]
    if !iszero(signal_ratio)
        noise = rand(Distributions.Normal(), num_points)
        noise .*= norm(y) / sqrt(signal_ratio) / norm(noise)
        y .+= noise
    end
    return (X, y)
end

function shapeconregr_JuMP(
    n::Int,
    deg::Int,
    num_points::Int,
    f::Function;
    signal_ratio::Float64 = 0.0,
    xmin::Float64 = -1.0,
    xmax::Float64 = 1.0,
    rseed::Int = 1,
    modeloptions...
    )
    (X, y) = generate_regr_data(n, num_points, f, signal_ratio = signal_ratio, xmin = xmin, xmax = xmax, rseed = rseed)
    return shapeconregr_JuMP(X, y, n, deg; modeloptions...)
end

function shapeconregr_JuMP(
    X::AbstractMatrix{Float64},
    y::AbstractVector{Float64},
    n::Int,
    deg::Int;
    use_lsq_obj::Bool = true,
    mono_dom::MU.Domain = MU.Box(-ones(n), ones(n)),
    conv_dom::MU.Domain = mono_dom,
    mono_profile::Vector{Int} = ones(Int, n),
    conv_profile::Int = 1,
    use_wsos::Bool = true,
    sample::Bool = true,
    rseed::Int = 1,
    )
    Random.seed!(rseed)
    num_points = size(X, 1)
    @assert n == MU.get_dimension(mono_dom) == MU.get_dimension(conv_dom) == size(X, 2)

    model = JuMP.Model()

    if use_wsos
        (regressor_points, _) = MU.get_interp_pts(MU.FreeDomain(n), deg, sample_factor = 50)
        lagrange_polys = MU.recover_lagrange_polys(regressor_points, deg)

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
        if !all(iszero, mono_profile)
            gradient_d = div(deg, 2)
            (mono_U, mono_points, mono_P0, mono_PWts, _) = MU.interpolate(mono_dom, gradient_d, sample = sample, sample_factor = 50)
            mono_wsos_cone = HYP.WSOSPolyInterpCone(mono_U, [mono_P0, mono_PWts...])
            for j in 1:n
                if !iszero(mono_profile[j])
                    gradient = DP.differentiate(regressor, DP.variables(regressor)[j])
                    JuMP.@constraint(model, [mono_profile[j] * gradient(mono_points[u, :]) for u in 1:mono_U] in mono_wsos_cone)
                end
            end
        end

        # convexity
        if !iszero(conv_profile)
            hessian_d = div(deg - 1, 2)
            (conv_U, conv_points, conv_P0, conv_PWts, _) = MU.interpolate(conv_dom, hessian_d, sample = sample, sample_factor = 50)
            conv_wsos_cone = HYP.WSOSPolyInterpMatCone(n, conv_U, [conv_P0, conv_PWts...])
            hessian = DP.differentiate(regressor, DP.variables(regressor), 2)
            JuMP.@constraint(model, [conv_profile * hessian[i, j](conv_points[u, :]) * (i == j ? 1.0 : rt2)
                for i in 1:n for j in 1:i for u in 1:conv_U] in conv_wsos_cone)
        end
    else
        PJ.setpolymodule!(model, SumOfSquares)
        d = div(deg + 1, 2)

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

        # monotonicity
        monotonic_set = MU.get_domain_inequalities(mono_dom, x)
        for j in 1:n
            if !iszero(mono_profile[j])
                gradient = DP.differentiate(p, x[j])
                JuMP.@constraint(model, mono_profile[j] * gradient >= 0, domain = monotonic_set)
            end
        end

        # convexity
        if !iszero(conv_profile)
            convex_set = MU.get_domain_inequalities(conv_dom, x)
            hessian = DP.differentiate(p, x, 2)
            JuMP.@constraint(model, conv_profile * hessian in JuMP.PSDCone(), domain = convex_set)
        end
    end

    return (model = model,)
end

shapeconregr1_JuMP() = shapeconregr_JuMP(2, 3, 100, x -> exp(norm(x)), use_lsq_obj = false)
shapeconregr2_JuMP() = shapeconregr_JuMP(2, 3, 100, x -> sum(x.^3), use_lsq_obj = false)
shapeconregr3_JuMP() = shapeconregr_JuMP(2, 3, 100, x -> sum(x.^4), use_lsq_obj = false)
shapeconregr4_JuMP() = shapeconregr_JuMP(2, 3, 100, x -> sum(x.^3), signal_ratio = 50.0, use_lsq_obj = false)
shapeconregr5_JuMP() = shapeconregr_JuMP(2, 3, 100, x -> sum(x.^4), signal_ratio = 50.0, use_lsq_obj = false)
shapeconregr6_JuMP() = shapeconregr_JuMP(2, 3, 100, x -> exp(norm(x)))
shapeconregr7_JuMP() = shapeconregr_JuMP(2, 3, 100, x -> sum(x.^4), signal_ratio = 50.0)
shapeconregr8_JuMP() = shapeconregr_JuMP(2, 4, 100, x -> -inv(1 + exp(-10.0 * norm(x))), mono_dom = MU.Box(zeros(2), ones(2)))
shapeconregr9_JuMP() = shapeconregr_JuMP(2, 4, 100, x -> -inv(1 + exp(-10.0 * norm(x))), signal_ratio = 10.0, mono_dom = MU.Box(zeros(2), ones(2)))
shapeconregr10_JuMP() = shapeconregr_JuMP(2, 4, 100, x -> exp(norm(x)))
shapeconregr11_JuMP() = shapeconregr_JuMP(2, 5, 100, x -> exp(norm(x)), signal_ratio = 10.0, mono_dom = MU.Box(0.5 * ones(2), 2 * ones(2)))
shapeconregr12_JuMP() = shapeconregr_JuMP(2, 6, 100, x -> exp(norm(x)), signal_ratio = 1.0, mono_dom = MU.Box(0.5 * ones(2), 2 * ones(2)), use_wsos = false)
shapeconregr13_JuMP() = shapeconregr_JuMP(2, 6, 100, x -> exp(norm(x)), signal_ratio = 1.0, use_lsq_obj = false)
shapeconregr14_JuMP() = shapeconregr_JuMP(5, 5, 1000, x -> exp(norm(x)), use_wsos = false)
shapeconregr15_JuMP() = shapeconregr_JuMP(2, 3, 100, x -> exp(norm(x)), use_lsq_obj = false, use_wsos = false)

function test_shapeconregr_JuMP(instance::Tuple{Function,Number}; options)
    (instance, true_obj) = instance
    data = instance()
    JuMP.optimize!(data.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(data.model) == MOI.OPTIMAL
    @test JuMP.objective_value(data.model) ≈ true_obj atol = 1e-4 rtol = 1e-4
    return
end

test_shapeconregr_JuMP(; options...) = test_shapeconregr_JuMP.([
    (shapeconregr1_JuMP, 4.4065e-1),
    (shapeconregr2_JuMP, 1.3971e-1),
    (shapeconregr3_JuMP, 2.4577e-1),
    (shapeconregr4_JuMP, 1.5449e-1),
    (shapeconregr5_JuMP, 2.5200e-1),
    (shapeconregr6_JuMP, 5.4584e-2),
    (shapeconregr7_JuMP, 3.3249e-2),
    (shapeconregr8_JuMP, 3.7723e-03),
    (shapeconregr9_JuMP, 3.0995e-02),
    (shapeconregr10_JuMP, 5.0209e-02),
    (shapeconregr11_JuMP, 0.22206),
    (shapeconregr12_JuMP, 0.22206),
    (shapeconregr13_JuMP, 1.7751), # not verified with SDP
    # (shapeconregr14_JuMP, NaN),
    (shapeconregr15_JuMP, 4.4065e-1),
    ], options = options)

test_shapeconregr_JuMP_small(; options...) = test_shapeconregr_JuMP.([
    (shapeconregr1_JuMP, 4.4065e-1),
    (shapeconregr2_JuMP, 1.3971e-1),
    (shapeconregr12_JuMP, 0.22206),
    (shapeconregr15_JuMP, 4.4065e-1),
    ], options = options)
