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

import LinearAlgebra: norm
import Random
import Distributions
using Test
import DataFrames
import CSV
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

function shapeconregrJuMP(
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
    )
    @assert n == MU.get_dimension(mono_dom) == MU.get_dimension(conv_dom) == size(X, 2)
    num_points = size(X, 1)

    if use_wsos
        (regressor_points, _) = MU.get_interp_pts(MU.FreeDomain(n), deg, sample_factor = 50)
        lagrange_polys = MU.recover_lagrange_polys(regressor_points, deg)

        model = JuMP.Model()
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
            gradient_halfdeg = div(deg, 2)
            (mono_U, mono_points, mono_P0, mono_PWts, _) = MU.interpolate(mono_dom, gradient_halfdeg, sample = sample, sample_factor = 50)
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
            hessian_halfdeg = div(deg - 1, 2)
            (conv_U, conv_points, conv_P0, conv_PWts, _) = MU.interpolate(conv_dom, hessian_halfdeg, sample = sample, sample_factor = 50)
            conv_wsos_cone = HYP.WSOSPolyInterpMatCone(n, conv_U, [conv_P0, conv_PWts...])
            hessian = DP.differentiate(regressor, DP.variables(regressor), 2)
            JuMP.@constraint(model, [conv_profile * hessian[i, j](conv_points[u, :]) * (i == j ? 1.0 : rt2) for i in 1:n for j in 1:i for u in 1:conv_U] in conv_wsos_cone)
        end
    else
        DP.@polyvar x[1:n]

        model = SumOfSquares.SOSModel()
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
                JuMP.@constraint(model, mono_profile[j] * gradient >= 0, domain = monotonic_set, maxdegree = 2 * div(deg, 2))
            end
        end

        # convexity
        if !iszero(conv_profile)
            convex_set = MU.get_domain_inequalities(conv_dom, x)
            hessian = DP.differentiate(p, x, 2)
            JuMP.@constraint(model, conv_profile * hessian in JuMP.PSDCone(), domain = convex_set, maxdegree = 2 * div(deg - 1, 2))
        end
    end

    return (model = model,)
end

function shapeconregrJuMP(
    n::Int,
    deg::Int,
    num_points::Int,
    f::Function;
    signal_ratio::Float64 = 0.0,
    xmin::Float64 = -1.0,
    xmax::Float64 = 1.0,
    model_kwargs...
    )
    X = rand(Distributions.Uniform(xmin, xmax), num_points, n)
    y = [f(X[p, :]) for p in 1:num_points]

    if !iszero(signal_ratio)
        noise = rand(Distributions.Normal(), num_points)
        noise .*= norm(y) / sqrt(signal_ratio) / norm(noise)
        y .+= noise
    end

    return shapeconregrJuMP(X, y, n, deg; model_kwargs...)
end




# see https://arxiv.org/pdf/1509.08165v1.pdf (example 1)
function normfunction_data(; n::Int = 5, num_points::Int = 100)
    f = x -> sum(abs2, x)
    (X, y) = generate_regr_data(n, num_points, f, 9.0, -1.0, 1.0)
    return (X, y, n)
end

# see https://arxiv.org/pdf/1509.08165v1.pdf (example 5)
function customfunction_data(; n::Int = 5, num_points::Int = 100)
    f = x -> (5x[1] + 0.5x[2] + x[3])^2 + sqrt(x[4]^2 + x[5]^2)
    (X, y) = generate_regr_data(n, num_points, f, 9.0, -1.0, 1.0)
    return (X, y, n)
end

# see https://arxiv.org/pdf/1509.08165v1.pdf (example 3)
# data obtained from http://www.nber.org/data/nbprod2005.html
function production_data()
    df = CSV.read(joinpath(@__DIR__, "data", "naics5811.csv"), copycols = true)
    deleterows!(df, 157) # outlier
    # number of non production employees
    df[:prode] .= df[:emp] - df[:prode]
    # group by industry codes
    dfagg = aggregate(dropmissing(df), :naics, sum)
    # four covariates: non production employees, production worker hours, production workers, total capital stock
    n = 4
    X = convert(Matrix, dfagg[[:prode_sum, :prodh_sum, :prodw_sum, :cap_sum]])
    # value of shipment
    y = convert(Array, dfagg[:vship_sum])
    # use the log transform of covariates
    Xlog = log.(X)
    # mean center
    Xlog .-= sum(Xlog, dims = 1) / size(Xlog, 1)
    y .-= sum(y) / length(y)
    # normalize to unit norm
    Xlog ./= sqrt.(sum(abs2, Xlog, dims = 1))
    y /= sqrt(sum(abs2, y))
    return (Xlog, y, n)
end

        (X, y, n) = s()

        (model,) = shapeconregrJuMP(X, y, n, deg, mono_dom = MU.FreeDomain(n), mono_profile = zeros(Int, n), conv_profile = 1)





shapeconregrJuMP1() = shapeconregrJuMP(2, 3, 100, x -> exp(norm(x)), use_lsq_obj = false)
shapeconregrJuMP2() = shapeconregrJuMP(2, 3, 100, x -> sum(x.^3), use_lsq_obj = false)
shapeconregrJuMP3() = shapeconregrJuMP(2, 3, 100, x -> sum(x.^4), use_lsq_obj = false)
shapeconregrJuMP4() = shapeconregrJuMP(2, 3, 100, x -> sum(x.^3), signal_ratio = 50.0, use_lsq_obj = false)
shapeconregrJuMP5() = shapeconregrJuMP(2, 3, 100, x -> sum(x.^4), signal_ratio = 50.0, use_lsq_obj = false)
shapeconregrJuMP6() = shapeconregrJuMP(2, 3, 100, x -> exp(norm(x)))
shapeconregrJuMP7() = shapeconregrJuMP(2, 3, 100, x -> sum(x.^4), signal_ratio = 50.0)
shapeconregrJuMP8() = shapeconregrJuMP(2, 4, 100, x -> -inv(1 + exp(-10.0 * norm(x))), mono_dom = MU.Box(zeros(2), ones(2)))
shapeconregrJuMP9() = shapeconregrJuMP(2, 4, 100, x -> -inv(1 + exp(-10.0 * norm(x))), signal_ratio = 10.0, mono_dom = MU.Box(zeros(2), ones(2)))
shapeconregrJuMP10() = shapeconregrJuMP(2, 4, 100, x -> exp(norm(x)))
shapeconregrJuMP11() = shapeconregrJuMP(2, 5, 100, x -> exp(norm(x)), signal_ratio = 10.0, mono_dom = MU.Box(0.5 * ones(2), 2 * ones(2)))
shapeconregrJuMP12() = shapeconregrJuMP(2, 6, 100, x -> exp(norm(x)), signal_ratio = 1.0, mono_dom = MU.Box(0.5 * ones(2), 2 * ones(2)), use_wsos = false)
shapeconregrJuMP13() = shapeconregrJuMP(2, 6, 100, x -> exp(norm(x)), signal_ratio = 1.0, use_lsq_obj = false)
shapeconregrJuMP14() = shapeconregrJuMP(5, 5, 1000, x -> exp(norm(x)), use_wsos = false)
shapeconregrJuMP15() = shapeconregrJuMP(2, 3, 100, x -> exp(norm(x)), use_lsq_obj = false, use_wsos = false)

function test_shapeconregrJuMP(instance::Tuple{Function, Number}; options, rseed::Int = 1)
    Random.seed!(rseed)
    (instance, true_obj) = instance
    data = instance()
    JuMP.optimize!(data.model, JuMP.with_optimizer(Hypatia.Optimizer; options...))
    @test JuMP.termination_status(data.model) == MOI.OPTIMAL
    @test JuMP.objective_value(data.model) ≈ true_obj atol = 1e-4 rtol = 1e-4
    return
end

# TODO remove the unknown objective values (try to instead compare objvals from pairs of formulations)
test_shapeconregrJuMP(; options...) = test_shapeconregrJuMP.([
    (shapeconregrJuMP1, 4.4065e-1),
    (shapeconregrJuMP2, 1.3971e-1),
    (shapeconregrJuMP3, 2.4577e-1),
    (shapeconregrJuMP4, 1.5449e-1),
    (shapeconregrJuMP5, 2.5200e-1),
    (shapeconregrJuMP6, 5.4584e-2),
    (shapeconregrJuMP7, 3.3249e-2),
    (shapeconregrJuMP8, 3.7723e-03),
    (shapeconregrJuMP9, 3.0995e-02),
    (shapeconregrJuMP10, 5.0209e-02),
    (shapeconregrJuMP11, 0.22206),
    (shapeconregrJuMP12, 0.22206),
    (shapeconregrJuMP13, 1.7751), # not verified with SDP
    # (shapeconregrJuMP14, NaN),
    (shapeconregrJuMP15, 4.4065e-1),
    ], options = options)

test_shapeconregrJuMP_quick(; options...) = test_shapeconregrJuMP.([
    (shapeconregrJuMP1, 4.4065e-1),
    (shapeconregrJuMP2, 1.3971e-1),
    (shapeconregrJuMP12, 0.22206),
    (shapeconregrJuMP15, 4.4065e-1),
    ], options = options)
