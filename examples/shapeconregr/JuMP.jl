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

using LinearAlgebra
import Random
using Test
import DelimitedFiles
import Distributions
import JuMP
const MOI = JuMP.MOI
import DynamicPolynomials
const DP = DynamicPolynomials
import PolyJuMP
import SumOfSquares
import MultivariateBases: FixedPolynomialBasis
import Hypatia
const MU = Hypatia.ModelUtilities

function shapeconregr_JuMP(
    T::Type{Float64}, # TODO support generic reals
    X::Matrix{Float64},
    y::Vector{Float64},
    deg::Int,
    use_wsos::Bool, # use WSOS cone formulation, else SDP formulation
    use_L1_obj::Bool, # in objective function use L1 norm, else L2 norm
    is_fit_exact::Bool;
    mono_dom::MU.Domain = MU.Box{Float64}(-ones(size(X, 2)), ones(size(X, 2))),
    conv_dom::MU.Domain = mono_dom,
    mono_profile::Vector{Int} = ones(Int, size(X, 2)),
    conv_profile::Int = 1,
    )
    n = size(X, 2)
    num_points = size(X, 1)

    (regressor_points, _) = MU.get_interp_pts(MU.FreeDomain{Float64}(n), deg, sample_factor = 50)
    lagrange_polys = MU.recover_lagrange_polys(regressor_points, deg)
    model = JuMP.Model()
    JuMP.@variable(model, regressor, PolyJuMP.Poly(FixedPolynomialBasis(lagrange_polys)))
    x = DP.variables(lagrange_polys)

    if use_wsos
        # monotonicity
        if !all(iszero, mono_profile)
            gradient_halfdeg = div(deg, 2)
            (mono_U, mono_points, mono_Ps, _) = MU.interpolate(mono_dom, gradient_halfdeg, sample = true, sample_factor = 50)
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
            (conv_U, conv_points, conv_Ps, _) = MU.interpolate(conv_dom, hessian_halfdeg, sample = true, sample_factor = 50)
            conv_wsos_cone = Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, conv_U, conv_Ps)
            hessian = DP.differentiate(regressor, x, 2)
            hessian_interp = [hessian[i, j](conv_points[u, :]) for i in 1:n for j in 1:i for u in 1:conv_U]
            MU.vec_to_svec!(hessian_interp, rt2 = sqrt(2), incr = conv_U)
            JuMP.@constraint(model, conv_profile * hessian_interp in conv_wsos_cone)
        end
    else
        SumOfSquares.setpolymodule!(model, SumOfSquares)

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

    return (model = model, is_fit_exact = is_fit_exact)
end

shapeconregr_JuMP(T::Type{Float64}, data_name::Symbol, args...; kwargs...) = shapeconregr_JuMP(T, eval(data_name)..., args...; kwargs...)

function shapeconregr_JuMP(
    T::Type{Float64}, # TODO support generic reals
    n::Int,
    num_points::Int,
    func::Symbol,
    signal_ratio::Real,
    args...;
    xmin::Real = -1,
    xmax::Real = 1,
    kwargs...
    )
    X = rand(Distributions.Uniform(xmin, xmax), num_points, n)
    f = shapeconregr_data[func]
    y = T[f(X[p, :]) for p in 1:num_points]
    if !iszero(signal_ratio)
        noise = randn(T, num_points)
        noise .*= norm(y) / sqrt(signal_ratio) / norm(noise)
        y .+= noise
    end
    return shapeconregr_JuMP(T, X, y, args...; kwargs...)
end

function test_shapeconregr_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = shapeconregr_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    r = d.model.moi_backend.optimizer.model.optimizer.result
    if d.is_fit_exact
        @test r.primal_obj ≈ 0 atol = 1e-4 rtol = 1e-4
    end
    return r
end

naics5811_data = begin
    Xy = DelimitedFiles.readdlm(joinpath(@__DIR__, "data", "naics5811.txt"))
    (Xy[:, 1:(end - 1)], Xy[:, end])
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
    :func9 => (x -> (5x[1] + 0.5x[2] + x[3])^2 + sqrt(x[4]^2 + x[5]^2)),
    )

shapeconregr_JuMP_fast = [
    (:naics5811_data, 4, true, false, false),
    (2, 50, :func1, 5, 3, true, false, false),
    (2, 50, :func2, 5, 3, true, true, false),
    (2, 50, :func3, 5, 3, false, false, false),
    (2, 50, :func4, 5, 3, false, true, false),
    (2, 50, :func5, 5, 4, true, false, false),
    (2, 50, :func6, 5, 4, true, true, false),
    (2, 50, :func7, 5, 4, false, false, false),
    (2, 50, :func8, 5, 4, false, true, false),
    (4, 150, :func6, 0, 4, true, false, true),
    (4, 150, :func6, 0, 4, true, true, true),
    (4, 150, :func7, 0, 4, true, false, true),
    (4, 150, :func7, 0, 4, true, true, true),
    (4, 150, :func7, 0, 4, false, false, true),
    (3, 150, :func8, 0, 6, true, false, true),
    (3, 150, :func8, 0, 6, true, true, true),
    (5, 100, :func9, 9, 4, true, false, false),
    ]
shapeconregr_JuMP_slow = [
    (4, 150, :func6, 0, 4, false, false, true),
    (3, 150, :func8, 0, 6, false, false, true),
    ]
