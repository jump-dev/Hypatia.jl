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

# a description of the shape of the regressor
mutable struct ShapeData
    mono_dom::MU.Domain
    conv_dom::MU.Domain
    mono_profile::Vector{Int}
    conv_profile::Int
end
ShapeData(n::Int) = ShapeData(MU.Box(-ones(n), ones(n)), MU.Box(-ones(n), ones(n)), ones(Int, n), 1)

function shapeconregr_JuMP(
    inst::Int;
    use_wsos::Bool = true,
    sample::Bool = true,
    signal_ratio::Float64 = 1.0,
    xmin::Float64 = -1.0,
    xmax::Float64 = 1.0,
    rseed::Int = 1,
    )
    Random.seed!(rseed)

    # TODO all this data should be in the shapeconregr_JuMP arguments
    (n, deg, num_points, signal_ratio, f, shape_data, use_lsq_obj, true_obj) = getshapeconregrdata(inst)

    X = rand(Distributions.Uniform(xmin, xmax), num_points, n)
    y = [f(X[p, :]) for p in 1:num_points]
    if !iszero(signal_ratio)
        noise = rand(Distributions.Normal(), num_points)
        noise .*= norm(y) / sqrt(signal_ratio) / norm(noise)
        y .+= noise
    end

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
        if !all(iszero, shape_data.mono_profile)
            gradient_d = div(deg, 2)
            (mono_U, mono_points, mono_P0, mono_PWts, _) = MU.interpolate(shape_data.mono_dom, gradient_d, sample = sample, sample_factor = 50)
            mono_wsos_cone = HYP.WSOSPolyInterpCone(mono_U, [mono_P0, mono_PWts...])
            for j in 1:n
                if !iszero(shape_data.mono_profile[j])
                    gradient = DP.differentiate(regressor, DP.variables(regressor)[j])
                    JuMP.@constraint(model, [shape_data.mono_profile[j] * gradient(mono_points[u, :]) for u in 1:mono_U] in mono_wsos_cone)
                end
            end
        end

        # convexity
        if !iszero(shape_data.conv_profile)
            hessian_d = div(deg - 1, 2)
            (conv_U, conv_points, conv_P0, conv_PWts, _) = MU.interpolate(shape_data.conv_dom, hessian_d, sample = sample, sample_factor = 50)
            conv_wsos_cone = HYP.WSOSPolyInterpMatCone(n, conv_U, [conv_P0, conv_PWts...])
            hessian = DP.differentiate(regressor, DP.variables(regressor), 2)
            JuMP.@constraint(model, [shape_data.conv_profile * hessian[i, j](conv_points[u, :]) * (i == j ? 1.0 : rt2)
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
        monotonic_set = MU.get_domain_inequalities(shape_data.mono_dom, x)
        for j in 1:n
            if !iszero(shape_data.mono_profile[j])
                gradient = DP.differentiate(p, x[j])
                JuMP.@constraint(model, shape_data.mono_profile[j] * gradient >= 0, domain = monotonic_set)
            end
        end

        # convexity
        if !iszero(shape_data.conv_profile)
            convex_set = MU.get_domain_inequalities(shape_data.conv_dom, x)
            hessian = DP.differentiate(p, x, 2)
            JuMP.@constraint(model, shape_data.conv_profile * hessian in JuMP.PSDCone(), domain = convex_set)
        end
    end

    return (model, true_obj)
end


# TODO remove duplicated variables and make into one-liner functions
function getshapeconregrdata(inst::Int)
    if inst == 1
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 0.0, x -> exp(norm(x)), false)
        shape_data = ShapeData(n)
        true_obj = 4.4065e-1
    elseif inst == 2
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 0.0, x -> sum(x.^3), false)
        shape_data = ShapeData(n)
        true_obj = 1.3971e-1
    elseif inst == 3
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 0.0, x -> sum(x.^4), false)
        shape_data = ShapeData(n)
        true_obj = 2.4577e-1
    elseif inst == 4
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 50.0, x -> sum(x.^3), false)
        shape_data = ShapeData(n)
        true_obj = 1.5449e-1
    elseif inst == 5
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 50.0, x -> sum(x.^4), false)
        shape_data = ShapeData(n)
        true_obj = 2.5200e-1
    elseif inst == 6
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 0.0, x -> exp(norm(x)), true)
        shape_data = ShapeData(n)
        true_obj = 5.4584e-2
    elseif inst == 7
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 50.0, x -> sum(x.^4), true)
        shape_data = ShapeData(n)
        true_obj = 3.3249e-2
    elseif inst == 8
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 4, 100, 0.0, x -> -inv(1 + exp(-10.0 * norm(x))), true)
        shape_data = ShapeData(MU.Box(zeros(n), ones(n)), MU.Box(zeros(n), ones(n)), ones(n), 1)
        true_obj = 3.7723e-03
    elseif inst == 9
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 4, 100, 10.0, x -> -inv(1 + exp(-10.0 * norm(x))), true)
        shape_data = ShapeData(MU.Box(zeros(n), ones(n)), MU.Box(zeros(n), ones(n)), ones(n), 1)
        true_obj = 3.0995e-02 # not verified with SDP
    elseif inst == 10
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 4, 100, 0.0, x -> exp(norm(x)), true)
        shape_data = ShapeData(n)
        true_obj = 5.0209e-02 # not verified with SDP
    elseif inst == 11
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 5, 100, 10.0, x -> exp(norm(x)), true)
        shape_data = ShapeData(MU.Box(0.5 * ones(n), 2 * ones(n)), MU.Box(0.5 * ones(n), 2 * ones(n)), ones(n), 1)
        true_obj = 0.22206 # not verified with SDP
    elseif inst == 12
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 6, 100, 1.0, x -> exp(norm(x)), true)
        shape_data = ShapeData(MU.Box(0.5 * ones(n), 2 * ones(n)), MU.Box(0.5 * ones(n), 2 * ones(n)), ones(n), 1)
        true_obj = 0.22206
    elseif inst == 13
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 6, 100, 1.0, x -> exp(norm(x)), false)
        shape_data = ShapeData(n)
        true_obj = 1.7751 # not verified with SDP
    elseif inst == 14
        # either out of memory error when converting sparse to dense in MOI conversion, or during preprocessing
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (5, 5, 1000, 0.0, x -> exp(norm(x)), true)
        shape_data = ShapeData(n)
        true_obj = NaN # unknown
    elseif inst == 15
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 0.0, x -> exp(norm(x)), false)
        shape_data = ShapeData(n)
        true_obj = 4.4065e-1
    else
        error("instance $inst not recognized")
    end
    return (n, deg, num_points, signal_ratio, f, shape_data, use_lsq_obj, true_obj)
end



shapeconregr1_JuMP() = shapeconregr_JuMP(1)
shapeconregr2_JuMP() = shapeconregr_JuMP(2)
shapeconregr3_JuMP() = shapeconregr_JuMP(3)
shapeconregr4_JuMP() = shapeconregr_JuMP(4)
shapeconregr5_JuMP() = shapeconregr_JuMP(5)
shapeconregr6_JuMP() = shapeconregr_JuMP(6)
shapeconregr7_JuMP() = shapeconregr_JuMP(7)
shapeconregr8_JuMP() = shapeconregr_JuMP(8)
shapeconregr9_JuMP() = shapeconregr_JuMP(9)
shapeconregr10_JuMP() = shapeconregr_JuMP(10)
shapeconregr11_JuMP() = shapeconregr_JuMP(11)
shapeconregr12_JuMP() = shapeconregr_JuMP(12, use_wsos = false)
shapeconregr13_JuMP() = shapeconregr_JuMP(13)
shapeconregr14_JuMP() = shapeconregr_JuMP(13, use_wsos = false)
shapeconregr15_JuMP() = shapeconregr_JuMP(15, use_PolyJuMP = true)

function test_shapeconregr_JuMP(instance::Function; options)
    (model, true_obj) = instance()
    JuMP.optimize!(model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.objective_value(model) ≈ true_obj atol = 1e-4 rtol = 1e-4
    return
end

test_shapeconregr_JuMP(; options...) = test_shapeconregr_JuMP.([
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
    ], options = options)

test_shapeconregr_JuMP_small(; options...) = test_shapeconregr_JuMP.([
    shapeconregr1_JuMP,
    shapeconregr2_JuMP,
    shapeconregr12_JuMP,
    shapeconregr15_JuMP,
    ], options = options)
