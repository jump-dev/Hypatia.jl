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
import MultivariateBases: FixedPolynomialBasis

struct ShapeConRegrJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    X::Matrix{T}
    y::Vector{T}
    deg::Int
    use_wsos::Bool # use WSOS cone formulation, else SDP formulation
    use_L1_obj::Bool # in objective function use L1 norm, else L2 norm
    use_monotonicity::Bool # if true add monotonicity constraints, else don't
    use_convexity::Bool # if true add convexity constraints, else don't
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
    Random.seed!(1) # TODO remove
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

example_tests(::Type{ShapeConRegrJuMP{Float64}}, ::MinimalInstances) = [
    ((:naics5811, 3, true, false, true, true, false),),
    ((:naics5811, 3, true, false, true, false, false),),
    ((:naics5811, 3, true, false, false, true, false),),
    ((1, 5, :func1, 2, 4, true, false, true, true, false),),
    ((1, 5, :func1, 2, 4, true, false, true, false, false),),
    ((1, 5, :func1, 2, 4, true, false, false, true, false),),
    ((1, 5, :func1, 2, 4, true, false, false, false, true),),
    ((1, 5, :func1, 2, 4, true, true, true, true, false),),
    ((1, 5, :func1, 2, 4, false, false, true, true, false),),
    ((1, 5, :func1, 2, 4, false, true, true, true, false), ClassicConeOptimizer),
    ((1, 5, :func1, 2, 4, false, true, true, true, false),),
    ((1, 5, :func1, 2, 4, false, true, true, true, false), ClassicConeOptimizer),
    ((1, 5, :func1, 2, 4, false, true, true, false, false), ClassicConeOptimizer),
    ((1, 5, :func1, 2, 4, false, true, false, true, false), ClassicConeOptimizer),
    ((1, 5, :func1, 2, 4, false, true, false, false, true), ClassicConeOptimizer),
    ]
example_tests(::Type{ShapeConRegrJuMP{Float64}}, ::FastInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6, verbose = true)
    relaxed_options = (tol_feas = 1e-4, tol_rel_opt = 1e-4, tol_abs_opt = 1e-4)
    return [
    # ((:naics5811, 4, true, false, true, true, false), nothing, options),
    # ((:naics5811, 4, true, true, true, true, false), nothing, relaxed_options),
    # ((:naics5811, 3, false, false, true, true, false), nothing, options),
    # ((:naics5811, 3, false, true, true, true, false), ClassicConeOptimizer, options),
    # ((:naics5811, 3, false, true, true, true, false), nothing, options),
    # ((:naics5811, 3, false, false, true, true, false), nothing, options),
    # ((:naics5811, 3, false, true, true, false, false), ClassicConeOptimizer, options),
    ((1, 100, :func1, 5, 25, true, false, false, true, false), nothing, options),
    ((1, 100, :func10, 5, 100, true, false, false, true, false), nothing, options),
    ((2, 100, :func1, 5, 10, true, false, false, true, false), nothing, options),
    # ((2, 50, :func1, 5, 3, true, false, true, true, false), nothing, options),
    # ((2, 50, :func1, 5, 3, true, false, true, false, false), nothing, options),
    # ((2, 50, :func1, 5, 3, true, false, false, true, false), nothing, options),
    # ((2, 200, :func1, 0, 3, true, false, false, false, true), nothing, options),
    # ((2, 50, :func2, 5, 3, true, true, true, true, false), nothing, options),
    # ((2, 50, :func3, 10, 3, false, true, false, true, false), nothing, options),
    # ((2, 50, :func3, 10, 3, true, true, false, true, false), nothing, options),
    #
    # ((2, 50, :func3, 5, 3, false, true, true, true, false), ClassicConeOptimizer, options),
    # ((2, 50, :func4, 5, 3, false, true, true, true, false), nothing, options),
    # ((2, 50, :func4, 5, 3, false, true, true, true, false), ClassicConeOptimizer, options),
    # ((2, 50, :func5, 5, 4, true, false, true, true, false), nothing, options),
    # ((2, 50, :func6, 5, 4, true, true, true, true, false), nothing, options),
    # ((2, 50, :func7, 5, 4, false, false, true, true, false), nothing, options),
    # ((2, 50, :func8, 5, 4, false, true, true, true, false), nothing, options),
    # ((4, 150, :func6, 0, 4, true, false, true, true, true), nothing, relaxed_options), # objective not tight enough
    # ((4, 150, :func7, 0, 4, true, false, true, true, true), nothing, options),
    # ((4, 150, :func7, 0, 4, true, true, true, true, true), nothing, options),
    # ((4, 150, :func7, 0, 4, false, false, true, true, true), nothing, options),
    # ((3, 150, :func8, 0, 6, true, false, true, true, true), nothing, relaxed_options),
    ]
end
example_tests(::Type{ShapeConRegrJuMP{Float64}}, ::SlowInstances) = begin
    options = (tol_feas = 1e-7, tol_rel_opt = 1e-6, tol_abs_opt = 1e-6)
    return [
    # ((:naics5811, 3, false, true, false, true, false), ClassicConeOptimizer, options),
    # ((:naics5811, 7, true, false, true, true, false), nothing, options),
    # ((:naics5811, 5, false, true, true, true, false), ClassicConeOptimizer, options),
    # ((4, 150, :func6, 0, 4, false, false, true, true, true), nothing, options),
    # ((4, 150, :func6, 0, 4, false, true, true, true, true), ClassicConeOptimizer, options),
    # ((3, 150, :func8, 0, 6, false, false, true, true, true), nothing, options),
    # ((3, 150, :func8, 0, 6, false, false, true, false, true), nothing, options),
    # ((3, 150, :func8, 0, 6, false, false, false, true, true), nothing, options),
    # ((3, 150, :func8, 0, 6, false, true, true, true, true), ClassicConeOptimizer, options),
    # ((3, 150, :func8, 0, 6, false, true, true, false, true), ClassicConeOptimizer, options),
    # ((3, 150, :func8, 0, 6, false, true, false, true, true), ClassicConeOptimizer, options),
    # ((5, 100, :func9, 9, 4, false, true, true, true, false), nothing, options),
    # ((5, 100, :func9, 9, 4, true, true, true, true, false), nothing, options),
    # ((5, 100, :func9, 9, 4, false, true, true, true, false), ClassicConeOptimizer, options),
    # ((5, 100, :func10, 4, 4, false, true, true, true, false), nothing, options),
    # ((5, 100, :func10, 4, 4, false, false, true, true, false), nothing, options),
    # ((5, 100, :func10, 4, 4, false, true, false, true, false), ClassicConeOptimizer, options),
    # ((5, 100, :func10, 4, 4, false, true, false, false, false), ClassicConeOptimizer, options),
    # ((5, 100, :func10, 4, 4, false, true, true, false, false), ClassicConeOptimizer, options),
    # ((5, 100, :func10, 4, 4, false, true, true, true, false), ClassicConeOptimizer, options),
    ]
end


function build(inst::ShapeConRegrJuMP{T}) where {T <: Float64} # TODO generic reals
    (X, y, deg) = (inst.X, inst.y, inst.deg)
    n = size(X, 2)
    num_points = size(X, 1)
    mono_dom = ModelUtilities.Box{T}(-ones(size(X, 2)), ones(size(X, 2)))
    conv_dom = mono_dom
    mono_profile = ones(Int, size(X, 2))
    conv_profile = 1
    U = binomial(n + deg, n)

    model = JuMP.Model()
    JuMP.@variable(model, regressor[1:U])

    (regressor_points, _) = ModelUtilities.get_interp_pts(ModelUtilities.FreeDomain{Float64}(n), deg)
    #
    # lagrange_polys = ModelUtilities.recover_lagrange_polys(regressor_points, deg)
    # regressor_fun = DP.polynomial(regressor, lagrange_polys)
    # x = DP.variables(lagrange_polys)
    #
    DP.@polyvar x[1:n]
    monos = DP.monomials(x, 0:deg)
    cheby_monos = basis_covering_monomials(ChebyshevBasisSecondKind, [monos...]).polynomials # yuck typing
    # vandermonde = [cheby_monos[j](view(regressor_points, i, :)) for i in 1:U, j in 1:U]
    vandermonde = [monos[j](view(regressor_points, i, :)) for i in 1:U, j in 1:U]
    @show cond(vandermonde)
    regressor_mono = inv(vandermonde) * regressor
    regressor_fun = DP.polynomial(regressor_mono, monos)
    # regressor_fun = DP.polynomial(regressor_mono, cheby_monos)

    # monotonicity
    if inst.use_monotonicity
        gradient_halfdeg = div(deg, 2)
        (mono_U, mono_points, mono_Ps, _) = ModelUtilities.interpolate(mono_dom, gradient_halfdeg)
        for j in 1:n
            if !iszero(mono_profile[j])
                gradient_fun = DP.differentiate(regressor_fun, x[j])
                gradient_interp = [gradient_fun(mono_points[u, :]) for u in 1:mono_U]
                if inst.use_wsos
                    JuMP.@constraint(model, mono_profile[j] .* gradient_interp in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(mono_U, mono_Ps))
                else
                    psd_vars = []
                    for (r, Pr) in enumerate(mono_Ps)
                        Lr = size(Pr, 2)
                        psd_r = JuMP.@variable(model, [1:Lr, 1:Lr], Symmetric)
                        push!(psd_vars, psd_r)
                        JuMP.@SDconstraint(model, mono_profile[j] .* psd_r >= 0)
                    end
                    coeffs_lhs = JuMP.@expression(model, [u in 1:mono_U], sum(sum(Pr[u, k] * Pr[u, l] * psd_r[k, l] * (k == l ? 1 : 2) for k in 1:size(Pr, 2) for l in 1:k) for (Pr, psd_r) in zip(mono_Ps, psd_vars)))
                    JuMP.@constraint(model, coeffs_lhs .== gradient_interp)
                end # use_wsos
            end # mono_profile
        end # j
    end # use_monotonicity

    # convexity
    if inst.use_convexity
        hessian_halfdeg = div(deg - 1, 2)
        (conv_U, conv_points, conv_Ps, _) = ModelUtilities.interpolate(conv_dom, hessian_halfdeg)
        hessian_fun = DP.differentiate(regressor_fun, x, 2)
        hessian_interp = [hessian_fun[i, j](conv_points[u, :]) for i in 1:n for j in 1:i for u in 1:conv_U]
        if inst.use_wsos
            ModelUtilities.vec_to_svec!(hessian_interp, rt2 = sqrt(2), incr = conv_U)
            JuMP.@constraint(model, conv_profile * hessian_interp in Hypatia.WSOSInterpPosSemidefTriCone{Float64}(n, conv_U, conv_Ps))
        else
            psd_vars = []
            for (r, Pr) in enumerate(conv_Ps)
                Lr = size(Pr, 2)
                psd_r = JuMP.@variable(model, [1:(Lr * n), 1:(Lr * n)], Symmetric)
                push!(psd_vars, psd_r)
                JuMP.@SDconstraint(model, conv_profile .* psd_r >= 0)
            end
            # for readability
            Ls = [size(Pr, 2) for Pr in conv_Ps]
            offset = 0
            for x1 in 1:n, x2 in 1:x1
                offset += 1
                # note that psd_vars[r][(x1 - 1) * Ls[r] + k, (x2 - 1) * Ls[r] + l] is not necessarily symmetric
                coeffs_lhs = JuMP.@expression(model, [u in 1:conv_U], sum(sum(conv_Ps[r][u, k] * conv_Ps[r][u, l] * psd_vars[r][(x1 - 1) * Ls[r] + k, (x2 - 1) * Ls[r] + l] for k in 1:Ls[r] for l in 1:Ls[r]) for r in eachindex(Ls)))
                JuMP.@constraint(model, coeffs_lhs .== hessian_interp[conv_U .* (offset - 1) .+ (1:conv_U)])
            end # x1, x2
        end # use_wsos
    end # use_convexity

    # objective function
    JuMP.@variable(model, z)
    JuMP.@objective(model, Min, z)
    norm_vec = [y[i] - regressor_fun(X[i, :]) for i in 1:num_points]

    if inst.use_L1_obj || (num_points <= U)
        obj_cone = (inst.use_L1_obj ? MOI.NormOneCone(1 + num_points) : MOI.SecondOrderCone(1 + num_points))
        JuMP.@constraint(model, vcat(z, norm_vec) in obj_cone)
    else
        # using L2 norm objective and number of samples exceeds variables, so use qr trick to reduce dimension
        coef_mat = zeros(num_points, U + 1)
        for (i, expr_i) in enumerate(norm_vec)
            for (c, v) in JuMP.linear_terms(expr_i)
                coef_mat[i, JuMP.index(v).value] = c
            end
            coef_mat[i, end] = JuMP.constant(expr_i)
        end
        coef_R = qr(coef_mat).R
        JuMP.@constraint(model, vcat(z, coef_R * vcat(regressor, 1)) in MOI.SecondOrderCone(2 + U))
    end

    return model
end

function test_extra(inst::ShapeConRegrJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    if JuMP.termination_status(model) == MOI.OPTIMAL && inst.is_fit_exact
        # check objective value is correct
        tol = eps(T)^0.25
        @test JuMP.objective_value(model) ≈ 0 atol = tol rtol = tol
    end
end

return ShapeConRegrJuMP
