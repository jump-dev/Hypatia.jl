#=
given data (xᵢ, yᵢ), find a polynomial p to solve
minimize    ∑ᵢℓ(p(xᵢ), yᵢ)
subject to  ρⱼ × dᵏp/dtⱼᵏ ≥ 0 ∀ t ∈ D
where
- dᵏp/dtⱼᵏ is the kᵗʰ derivative of p in direction j,
- ρⱼ determines the desired sign of the derivative,
- D is a domain such as a box or an ellipsoid,
- ℓ is a convex loss function.
see e.g. Chapter 8 of thesis by G. Hall (2018)
=#

import DelimitedFiles
import Distributions

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
    # TODO assert X data is on the domain [-1, 1]^n
    return ShapeConRegrJuMP{Float64}(X, y, args...)
end

function ShapeConRegrJuMP{Float64}(
    n::Int,
    num_points::Int,
    func::Symbol,
    signal_ratio::Real,
    args...)
    X = rand(Distributions.Uniform(-1, 1), num_points, n)
    f = shapeconregr_data[func]
    y = Float64[f(X[p, :]) for p in 1:num_points]
    if !iszero(signal_ratio)
        noise = randn(num_points)
        noise .*= norm(y) / sqrt(signal_ratio) / norm(noise)
        y .+= noise
    end
    return ShapeConRegrJuMP{Float64}(X, y, args...)
end

function build(inst::ShapeConRegrJuMP{T}) where {T <: Float64}
    (X, y, deg) = (inst.X, inst.y, inst.deg)
    n = size(X, 2)
    num_points = size(X, 1)
    mono_dom = PolyUtils.BoxDomain{T}(-ones(size(X, 2)), ones(size(X, 2)))
    conv_dom = mono_dom
    mono_profile = ones(Int, size(X, 2))
    conv_profile = 1

    # setup interpolation (not actually using FreeDomain, just need points here)
    halfdeg = div(deg + 1, 2)
    free_dom = PolyUtils.FreeDomain{T}(n)
    (U, points, Ps, V) = PolyUtils.interpolate(free_dom,
        halfdeg, calc_V = true)
    F = qr!(Array(V'), ColumnNorm()) # TODO reuse QR parts
    V_X = PolyUtils.make_chebyshev_vandermonde(X, 2halfdeg)
    X_points_polys = F \ V_X'

    model = JuMP.Model()
    JuMP.@variable(model, regressor[1:U])
    JuMP.@variable(model, z)
    JuMP.@objective(model, Min, z)

    # objective epigraph
    norm_vec = y - X_points_polys' * regressor
    if inst.use_L1_obj || (num_points <= U)
        obj_K = (inst.use_L1_obj ? MOI.NormOneCone : MOI.SecondOrderCone)(
            1 + num_points)
        JuMP.@constraint(model, vcat(z, norm_vec) in obj_K)
    else
        # using L2 norm objective and number of samples exceeds variables,
        # so use qr trick to reduce dimension
        coef_mat = zeros(num_points, U + 1)
        for (i, expr_i) in enumerate(norm_vec)
            for (c, v) in JuMP.linear_terms(expr_i)
                coef_mat[i, JuMP.index(v).value] = c
            end
            coef_mat[i, end] = JuMP.constant(expr_i)
        end
        coef_R = qr(coef_mat).R
        JuMP.@constraint(model, vcat(z, coef_R * vcat(regressor, 1)) in
            MOI.SecondOrderCone(2 + U))
    end

    # monotonicity
    if inst.use_monotonicity
        gradient_halfdeg = div(deg, 2)
        (mono_U, mono_points, mono_Ps) =
            PolyUtils.interpolate(mono_dom, gradient_halfdeg)
        univ_chebs_der = [PolyUtils.calc_univariate_chebyshev(
            mono_points[:, i], 2halfdeg, calc_gradient = true) for i in 1:n]

        for j in 1:n
            iszero(mono_profile[j]) && continue

            univ_chebs_g = [univ_chebs_der[i][(i == j) ? 2 : 1] for i in 1:n]
            V_g = PolyUtils.make_product_vandermonde(univ_chebs_g,
                PolyUtils.n_deg_exponents(n, 2halfdeg))
            scal = inv(maximum(abs, V_g) / 10)
            if scal < 1e-7
                @warn("model is numerically challenging to set up", maxlog = 1)
            end
            lmul!(scal, V_g)
            g_points_polys = F \ V_g'

            gradient_interp = mono_profile[j] * g_points_polys' * regressor

            if inst.use_wsos
                JuMP.@constraint(model, gradient_interp in
                    Hypatia.WSOSInterpNonnegativeCone{T, T}(mono_U, mono_Ps))
            else
                psd_vars = []
                for (r, Pr) in enumerate(mono_Ps)
                    Lr = size(Pr, 2)
                    psd_r = JuMP.@variable(model, [1:Lr, 1:Lr], Symmetric)
                    push!(psd_vars, psd_r)
                    JuMP.@SDconstraint(model, psd_r >= 0)
                end
                coeffs_lhs = JuMP.@expression(model, [u in 1:mono_U],
                    sum(sum(Pr[u, k] * Pr[u, l] * psd_r[k, l] * (k == l ? 1 : 2)
                    for k in 1:size(Pr, 2) for l in 1:k)
                    for (Pr, psd_r) in zip(mono_Ps, psd_vars)))
                JuMP.@constraint(model, coeffs_lhs .== gradient_interp)
            end
        end
    end

    # convexity
    if inst.use_convexity && !iszero(conv_profile)
        hessian_halfdeg = div(deg - 1, 2)
        (conv_U, conv_points, conv_Ps) =
            PolyUtils.interpolate(conv_dom, hessian_halfdeg)
        univ_chebs_der = [PolyUtils.calc_univariate_chebyshev(
            conv_points[:, i], 2halfdeg, calc_gradient = true,
            calc_hessian = true) for i in 1:n]

        deriv_num(i, j, k) = (k != i && k != j && return 1;
            k == i && k == j && return 3; return 2)

        V_Hs = Matrix{T}[]
        for i in 1:n, j in 1:i
            univ_chebs_H = [univ_chebs_der[k][deriv_num(i, j, k)] for k in 1:n]
            V_H = PolyUtils.make_product_vandermonde(univ_chebs_H,
                PolyUtils.n_deg_exponents(n, 2halfdeg))
            push!(V_Hs, V_H)
        end
        scal = inv(maximum(maximum(abs, V_H) for V_H in V_Hs))
        if scal < 1e-7
            @warn("model is numerically challenging to set up", maxlog = 1)
        end
        lmul!.(scal, V_Hs)

        H_interp = conv_profile * vcat([(F \ V_H')' *
            regressor for V_H in V_Hs]...)

        if inst.use_wsos
            if n == 1
                conv_K = Hypatia.WSOSInterpNonnegativeCone{T, T}(conv_U, conv_Ps)
            else
                Cones.scale_svec!(H_interp, sqrt(T(2)), incr = conv_U)
                conv_K = Hypatia.WSOSInterpPosSemidefTriCone{T}(n, conv_U, conv_Ps)
            end
            JuMP.@constraint(model, H_interp in conv_K)
        else
            psd_vars = []
            for (r, Pr) in enumerate(conv_Ps)
                Lr = size(Pr, 2)
                psd_r = JuMP.@variable(model, [1:(Lr * n), 1:(Lr * n)], Symmetric)
                push!(psd_vars, psd_r)
                JuMP.@SDconstraint(model, psd_r >= 0)
            end
            # for readability
            Ls = [size(Pr, 2) for Pr in conv_Ps]
            offset = 0
            for x1 in 1:n, x2 in 1:x1
                offset += 1
                coeffs_lhs = JuMP.@expression(model, [u in 1:conv_U],
                    sum(sum(conv_Ps[r][u, k] * conv_Ps[r][u, l] *
                    psd_vars[r][(x1 - 1) * Ls[r] + k, (x2 - 1) * Ls[r] + l]
                    for k in 1:Ls[r] for l in 1:Ls[r]) for r in eachindex(Ls)))
                JuMP.@constraint(model, coeffs_lhs .==
                    H_interp[conv_U .* (offset - 1) .+ (1:conv_U)])
            end
        end
    end

    return model
end

function test_extra(inst::ShapeConRegrJuMP{T}, model::JuMP.Model) where T
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    if (stat == MOI.OPTIMAL) && inst.is_fit_exact
        # check objective value is correct
        tol = eps(T)^0.25
        @test JuMP.objective_value(model) ≈ 0 atol=tol rtol=tol
    end
end

shapeconregr_data = Dict(
    :func1 => (x -> sum(x .^ 2)),
    :func2 => (x -> sum(x .^ 3)),
    :func3 => (x -> sum(x .^ 4)),
    :func4 => (x -> exp(norm(x)^2 / length(x)) - 1),
    :func5 => (x -> -inv(1 + exp(-10 * norm(x)))),
    :func6 => (x -> sum((x .+ 1) .^ 4)),
    :func7 => (x -> sum((x / 2 .+ 1) .^ 3)),
    :func8 => (x -> sum((x .+ 1) .^ 5 .- 2)),
    :func9 => (x -> (5x[1] + x[2] / 2 + x[3])^2 + sqrt(x[4]^2 + x[5]^2)),
    :func10 => (x -> sum(exp.(x))),
    )
