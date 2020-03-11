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

include(joinpath(@__DIR__, "../common_JuMP.jl"))
import DelimitedFiles
import Distributions
import DynamicPolynomials
const DP = DynamicPolynomials
import PolyJuMP
import SumOfSquares
import MultivariateBases: FixedPolynomialBasis

struct ShapeConRegrJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    X::Matrix{T}
    y::Vector{T}
    deg::Int
    use_wsos::Bool # use WSOS cone formulation, else SDP formulation
    use_L1_obj::Bool # in objective function use L1 norm, else L2 norm
    is_fit_exact::Bool
end
function ShapeConRegrJuMP{Float64}(
    data_name::Symbol,
    args...)
    Xy = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "$data_name.txt"))
    (X, y) = (Xy[:, 1:(end - 1)], Xy[:, end])
    return ShapeConRegrJuMP{Float64}(X, y, args...)
end
function ShapeConRegrJuMP{Float64}(
    n::Int,
    num_points::Int,
    func::Symbol,
    signal_ratio::Real,
    args...;
    xmin::Real = -1,
    xmax::Real = 1)
    X = rand(Distributions.Uniform(xmin, xmax), num_points, n)
    f = shapeconregr_data[func]
    y = Float64[f(X[p, :]) for p in 1:num_points]
    if !iszero(signal_ratio)
        noise = randn(num_points)
        noise .*= norm(y) / sqrt(signal_ratio) / norm(noise)
        y .+= noise
    end
    return ShapeConRegrJuMP{Float64}(X, y, args...)
end

shapeconregr_data = Dict(
    :func1 => (x -> sum(x .^ 2)),
    :func2 => (x -> sum(x .^ 3)),
    :func3 => (x -> sum(x .^ 4)),
    :func4 => (x -> exp(norm(x))),
    :func5 => (x -> -inv(1 + exp(-10 * norm(x)))),
    :func6 => (x -> sum((x .+ 1) .^ 4)),
    :func7 => (x -> sum((x / 2 .+ 1) .^ 3)),
    :func8 => (x -> sum((x .+ 1) .^ 5 .- 2)),
    :func9 => (x -> (5x[1] + x[2] / 2 + x[3])^2 + sqrt(x[4]^2 + x[5]^2)),
    :func10 => (x -> sum(exp.(x))),
    )

options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
example_tests(::Type{ShapeConRegrJuMP{Float64}}, ::MinimalInstances) = [
    ((:naics5811, 4, true, false, false), false, options),
    ((1, 5, :func1, 2, 4, true, false, false), false, options),
    ]
example_tests(::Type{ShapeConRegrJuMP{Float64}}, ::FastInstances) = [
    ((:naics5811, 6, true, false, false), false, options),
    ((2, 50, :func1, 5, 3, true, false, false), false, options),
    ((2, 50, :func2, 5, 3, true, true, false), false, options),
    ((2, 50, :func3, 5, 3, false, false, false), false, options),
    ((2, 50, :func4, 5, 3, false, true, false), false, options),
    ((2, 50, :func5, 5, 4, true, false, false), false, options),
    ((2, 50, :func6, 5, 4, true, true, false), false, options),
    ((2, 50, :func7, 5, 4, false, false, false), false, options),
    ((2, 50, :func8, 5, 4, false, true, false), false, options),
    ((4, 150, :func6, 0, 4, true, false, true), false, options),
    ((4, 150, :func6, 0, 4, true, true, true), false, options),
    ((4, 150, :func7, 0, 4, true, false, true), false, options),
    ((4, 150, :func7, 0, 4, true, true, true), false, options),
    ((4, 150, :func7, 0, 4, false, false, true), false, options),
    ((3, 150, :func8, 0, 6, true, false, true), false, options),
    ((3, 150, :func8, 0, 6, true, true, true), false, options),
    ((5, 100, :func9, 9, 4, true, false, false), false, options),
    ((5, 100, :func10, 4, 4, true, false, false), false, options),
    ]
example_tests(::Type{ShapeConRegrJuMP{Float64}}, ::SlowInstances) = [
    ((:naics5811, 8, true, false, false), false, options),
    ((4, 150, :func6, 0, 4, false, false, true), false, options),
    ((3, 150, :func8, 0, 6, false, false, true), false, options),
    ]

function build(inst::ShapeConRegrJuMP{T}) where {T <: Float64} # TODO generic reals
    (X, y, deg) = (inst.X, inst.y, inst.deg)
    n = size(X, 2)
    num_points = size(X, 1)
    # TODO allow options for below
    mono_dom = ModelUtilities.Box{T}(-ones(size(X, 2)), ones(size(X, 2)))
    conv_dom = mono_dom
    mono_profile = ones(Int, size(X, 2))
    conv_profile = 1

    (regressor_points, _) = ModelUtilities.get_interp_pts(ModelUtilities.FreeDomain{Float64}(n), deg)
    lagrange_polys = ModelUtilities.recover_lagrange_polys(regressor_points, deg)

    model = JuMP.Model()
    JuMP.@variable(model, regressor, PolyJuMP.Poly(FixedPolynomialBasis(lagrange_polys)))
    x = DP.variables(lagrange_polys)

    if inst.use_wsos
        # monotonicity
        if !all(iszero, mono_profile)
            gradient_halfdeg = div(deg, 2)
            (mono_U, mono_points, mono_Ps, _) = ModelUtilities.interpolate(mono_dom, gradient_halfdeg)
            mono_wsos_cone = Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(mono_U, mono_Ps)
            for j in 1:n
                if !iszero(mono_profile[j])
                    gradient = DP.differentiate(regressor, x[j])
                    JuMP.@constraint(model, [mono_profile[j] * gradient(mono_points[u, :]) for u in 1:mono_U] in mono_wsos_cone)
                end
            end
        end

        # convexity
        if !iszero(conv_profile)
            hessian_halfdeg = div(deg - 1, 2)
            (conv_U, conv_points, conv_Ps, _) = ModelUtilities.interpolate(conv_dom, hessian_halfdeg)
            conv_wsos_cone = Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, conv_U, conv_Ps)
            hessian = DP.differentiate(regressor, x, 2)
            hessian_interp = [hessian[i, j](conv_points[u, :]) for i in 1:n for j in 1:i for u in 1:conv_U]
            ModelUtilities.vec_to_svec!(hessian_interp, rt2 = sqrt(2), incr = conv_U)
            JuMP.@constraint(model, conv_profile * hessian_interp in conv_wsos_cone)
        end
    else
        PolyJuMP.setpolymodule!(model, SumOfSquares)

        # monotonicity
        monotonic_set = ModelUtilities.get_domain_inequalities(mono_dom, x)
        for j in 1:n
            if !iszero(mono_profile[j])
                gradient = DP.differentiate(regressor, x[j])
                JuMP.@constraint(model, mono_profile[j] * gradient >= 0, domain = monotonic_set, maxdegree = 2 * div(deg, 2))
            end
        end

        # convexity
        if !iszero(conv_profile)
            convex_set = ModelUtilities.get_domain_inequalities(conv_dom, x)
            hessian = DP.differentiate(regressor, x, 2)
            # maxdegree of each element in the SOS-matrix is 2 * div(deg - 1, 2), but we add 2 to take auxiliary monomials into account from the SumOfSquares transformation
            JuMP.@constraint(model, Symmetric(conv_profile * hessian) in JuMP.PSDCone(), domain = convex_set, maxdegree = 2 * div(deg - 1, 2) + 2)
        end
    end

    # objective function
    variables = JuMP.all_variables(model)
    num_vars = length(variables)
    @assert num_vars == DP.nterms(regressor)
    JuMP.@variable(model, z)
    JuMP.@objective(model, Min, z)
    norm_vec = [y[i] - regressor(X[i, :]) for i in 1:num_points]

    if inst.use_L1_obj || (num_points <= num_vars)
        obj_cone = (inst.use_L1_obj ? MOI.NormOneCone(1 + num_points) : MOI.SecondOrderCone(1 + num_points))
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

    return model
end

function test_extra(inst::ShapeConRegrJuMP, model, options)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    if JuMP.termination_status(model) == MOI.OPTIMAL && inst.is_fit_exact
        # check objective value is correct
        tol = eps(T)^0.25
        @test JuMP.objective_value(model) ≈ 0 atol = tol rtol = tol
    end
end

# @testset "ShapeConRegrJuMP" for inst in example_tests(ShapeConRegrJuMP{Float64}, MinimalInstances()) test(inst...) end

return ShapeConRegrJuMP
