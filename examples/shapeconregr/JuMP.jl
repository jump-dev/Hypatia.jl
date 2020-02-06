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

TODO
- reduce dimension in case of more observations by doing cholesky trick, as in matrixregression example
- ? for odd degrees WSOS/PSD formulations don't match, modeling issue
=#

using LinearAlgebra
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
import MultivariateBases: FixedPolynomialBasis
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia
const MU = Hypatia.ModelUtilities

const rt2 = sqrt(2)

function shapeconregrJuMP(
    X::Matrix{Float64},
    y::Vector{Float64},
    deg::Int;
    n::Int = size(X, 2),
    use_wsos::Bool = true, # use WSOS cone formulation, else SDP formulation
    use_L1_obj::Bool = false, # in objective function use L1 norm, else L2 norm
    sample::Bool = true,
    mono_dom::MU.Domain = MU.Box{Float64}(-ones(n), ones(n)),
    conv_dom::MU.Domain = mono_dom,
    mono_profile::Vector{Int} = ones(Int, n),
    conv_profile::Int = 1,
    )
    @assert n == size(X, 2)
    num_points = size(X, 1)

    if use_wsos
        (regressor_points, _) = MU.get_interp_pts(MU.FreeDomain{Float64}(n), deg, sample_factor = 50)
        lagrange_polys = MU.recover_lagrange_polys(regressor_points, deg)

        model = JuMP.Model()
        JuMP.@variable(model, regressor, variable_type = PJ.Poly(FixedPolynomialBasis(lagrange_polys)))

        # monotonicity
        if !all(iszero, mono_profile)
            gradient_halfdeg = div(deg, 2)
            (mono_U, mono_points, mono_Ps, _) = MU.interpolate(mono_dom, gradient_halfdeg, sample = sample, sample_factor = 50)
            mono_wsos_cone = Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(mono_U, mono_Ps)
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
            (conv_U, conv_points, conv_Ps, _) = MU.interpolate(conv_dom, hessian_halfdeg, sample = sample, sample_factor = 50)
            conv_wsos_cone = Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, conv_U, conv_Ps)
            hessian = DP.differentiate(regressor, DP.variables(regressor), 2)
            hessian_interp = [hessian[i, j](conv_points[u, :]) for i in 1:n for j in 1:i for u in 1:conv_U]
            MU.vec_to_svec!(hessian_interp, rt2 = sqrt(2), incr = conv_U)
            JuMP.@constraint(model, conv_profile * hessian_interp in conv_wsos_cone)
        end
    else
        DP.@polyvar x[1:n]

        model = SumOfSquares.SOSModel()
        JuMP.@variable(model, regressor, PJ.Poly(DP.monomials(x, 0:deg)))

        # monotonicity
        monotonic_set = MU.get_domain_inequalities(mono_dom, x)
        for j in 1:n
            if !iszero(mono_profile[j])
                gradient = DP.differentiate(regressor, x[j])
                JuMP.@constraint(model, mono_profile[j] * gradient >= 0, domain = monotonic_set, maxdegree = 2 * div(deg, 2))
            end
        end

        # convexity
        if !iszero(conv_profile)
            convex_set = MU.get_domain_inequalities(conv_dom, x)
            hessian = DP.differentiate(regressor, x, 2)
            JuMP.@constraint(model, conv_profile * hessian in JuMP.PSDCone(), domain = convex_set, maxdegree = 2 * div(deg - 1, 2))
        end
    end

    # objective function
    variables = JuMP.all_variables(model)
    num_vars = length(variables)
    @assert num_vars == DP.nterms(regressor)
    JuMP.@variable(model, z)
    JuMP.@objective(model, Min, z)
    norm_vec = [y[i] - regressor(X[i, :]) for i in 1:num_points]
    if use_L1_obj || (num_points <= num_vars)
        obj_cone = (use_L1_obj ? MOI.NormOneCone(1 + num_points) : MOI.SecondOrderCone(1 + num_points))
        JuMP.@constraint(model, vcat(z, norm_vec) in obj_cone)
    else
        # using L2 norm objective and number of samples exceeds variables, so use qr trick to reduce dimension
        coef_mat = zeros(num_points, num_vars + 1)
        for (i, expr_i) in enumerate(norm_vec)
            for (c, v) in JuMP.linear_terms(expr_i)
                coef_mat[i, JuMP.index(v).value] = c
            end
            coef_mat[i, end] = JuMP.constant(expr_i)
        end
        coef_R = qr(coef_mat).R
        JuMP.@constraint(model, vcat(z, coef_R * vcat(variables, 1)) in MOI.SecondOrderCone(2 + num_vars))
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
        noise = randn(num_points)
        noise .*= norm(y) / sqrt(signal_ratio) / norm(noise)
        y .+= noise
    end

    return shapeconregrJuMP(X, y, deg; model_kwargs...)
end

# see https://arxiv.org/pdf/1509.08165v1.pdf (example 3)
# data obtained from http://www.nber.org/data/nbprod2005.html
function production_data()
    df = CSV.read(joinpath(@__DIR__, "data", "naics5811.csv"), copycols = true)
    DataFrames.deleterows!(df, 157) # outlier
    # number of non production employees
    df[!, :prode] .= df[!, :emp] - df[!, :prode]
    # group by industry codes
    df_aggr = DataFrames.aggregate(DataFrames.dropmissing(df), :naics, sum)

    # four covariates: non production employees, production worker hours, production workers, total capital stock
    # use the log transform of covariates
    X = log.(convert(Matrix{Float64}, df_aggr[!, [:prode_sum, :prodh_sum, :prodw_sum, :cap_sum]])) # n = 4
    # value of shipment
    y = convert(Vector{Float64}, df_aggr[!, :vship_sum])
    # mean center
    X .-= sum(X, dims = 1) ./ size(X, 1)
    y .-= sum(y) / length(y)
    # normalize to unit norm
    X ./= norm.(eachcol(X))'
    y /= norm(y)

    return (X, y)
end

shapeconregrJuMP1() = shapeconregrJuMP(production_data()..., 4, mono_dom = MU.FreeDomain{Float64}(4), mono_profile = zeros(Int, 4))
shapeconregrJuMP2() = shapeconregrJuMP(2, 3, 100, x -> sum(x .^ 3), use_L1_obj = true)
shapeconregrJuMP3() = shapeconregrJuMP(2, 3, 100, x -> sum(x .^ 4), use_L1_obj = true)
shapeconregrJuMP4() = shapeconregrJuMP(2, 3, 100, x -> sum(x .^ 3), signal_ratio = 50.0, use_L1_obj = true)
shapeconregrJuMP5() = shapeconregrJuMP(2, 3, 100, x -> sum(x .^ 4), signal_ratio = 50.0, use_L1_obj = true)
shapeconregrJuMP6() = shapeconregrJuMP(2, 3, 100, x -> exp(norm(x)))
shapeconregrJuMP7() = shapeconregrJuMP(2, 3, 100, x -> sum(x .^ 4), signal_ratio = 50.0)
shapeconregrJuMP8() = shapeconregrJuMP(2, 4, 100, x -> -inv(1 + exp(-10.0 * norm(x))), mono_dom = MU.Box{Float64}(zeros(2), ones(2)))
shapeconregrJuMP9() = shapeconregrJuMP(2, 4, 100, x -> -inv(1 + exp(-10.0 * norm(x))), signal_ratio = 10.0, mono_dom = MU.Box{Float64}(zeros(2), ones(2)))
shapeconregrJuMP10() = shapeconregrJuMP(2, 4, 100, x -> exp(norm(x)))
shapeconregrJuMP11() = shapeconregrJuMP(2, 5, 100, x -> exp(norm(x)), signal_ratio = 10.0, mono_dom = MU.Box{Float64}(0.5 * ones(2), 2 * ones(2)))
shapeconregrJuMP12() = shapeconregrJuMP(2, 5, 100, x -> exp(norm(x)), signal_ratio = 1.0, mono_dom = MU.Box{Float64}(0.5 * ones(2), 2 * ones(2)), use_wsos = false)
shapeconregrJuMP13() = shapeconregrJuMP(2, 6, 100, x -> exp(norm(x)), signal_ratio = 1.0, use_L1_obj = true)
shapeconregrJuMP14() = shapeconregrJuMP(5, 2, 50, x -> exp(norm(x)), use_L1_obj = true, use_wsos = false)
shapeconregrJuMP15() = shapeconregrJuMP(2, 3, 100, x -> exp(norm(x)), use_L1_obj = true, use_wsos = false)
shapeconregrJuMP16() = shapeconregrJuMP(5, 3, 100, x -> sum(x .^ 2), signal_ratio = 9.0) # see https://arxiv.org/pdf/1509.08165v1.pdf (example 1)
shapeconregrJuMP17() = shapeconregrJuMP(5, 4, 100, x -> (5x[1] + 0.5x[2] + x[3])^2 + sqrt(x[4]^2 + x[5]^2), signal_ratio = 9.0) # see https://arxiv.org/pdf/1509.08165v1.pdf (example 5)
shapeconregrJuMP18() = shapeconregrJuMP(2, 4, 100, x -> sum((x .+ 1) .^ 4), signal_ratio = 0.0)
shapeconregrJuMP19() = shapeconregrJuMP(3, 4, 100, x -> sum((0.5 * x .+ 1) .^ 3), signal_ratio = 0.0)
shapeconregrJuMP20() = shapeconregrJuMP(3, 6, 100, x -> sum((x .+ 1) .^ 5 .- 2), signal_ratio = 0.0)

function test_shapeconregrJuMP(instance::Tuple{Function, Number}; options, rseed::Int = 1)
    Random.seed!(rseed)
    (instance, true_obj) = instance
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    if !isnan(true_obj)
        @test JuMP.objective_value(d.model) ≈ true_obj atol = 1e-4 rtol = 1e-4
    end
    return
end

test_shapeconregrJuMP_all(; options...) = test_shapeconregrJuMP.([
    (shapeconregrJuMP1, NaN),
    (shapeconregrJuMP2, NaN),
    (shapeconregrJuMP3, NaN),
    (shapeconregrJuMP4, NaN),
    (shapeconregrJuMP5, NaN),
    (shapeconregrJuMP6, NaN),
    (shapeconregrJuMP7, NaN),
    (shapeconregrJuMP8, NaN),
    (shapeconregrJuMP9, NaN),
    (shapeconregrJuMP10, NaN),
    (shapeconregrJuMP11, NaN),
    (shapeconregrJuMP12, NaN),
    (shapeconregrJuMP13, NaN),
    (shapeconregrJuMP14, NaN),
    (shapeconregrJuMP15, NaN),
    (shapeconregrJuMP16, NaN),
    (shapeconregrJuMP17, NaN),
    (shapeconregrJuMP18, 0.0),
    (shapeconregrJuMP19, 0.0),
    (shapeconregrJuMP20, 0.0),
    ], options = options)

test_shapeconregrJuMP(; options...) = test_shapeconregrJuMP.([
    (shapeconregrJuMP1, NaN),
    (shapeconregrJuMP2, NaN),
    (shapeconregrJuMP4, NaN),
    (shapeconregrJuMP6, NaN),
    (shapeconregrJuMP12, NaN),
    (shapeconregrJuMP15, NaN),
    (shapeconregrJuMP16, NaN),
    (shapeconregrJuMP17, NaN),
    (shapeconregrJuMP18, 0.0),
    (shapeconregrJuMP19, 0.0),
    (shapeconregrJuMP20, 0.0),
    ], options = options)
