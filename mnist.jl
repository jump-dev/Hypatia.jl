using Random
using MLDatasets
using Hypatia
import Hypatia.ModelUtilities: FreeDomain, interpolate, calc_univariate_chebyshev,
    make_product_vandermonde, make_chebyshev_vandermonde, n_deg_exponents
using JuMP
using DynamicPolynomials
# using TypedPolynomials
using PolyJuMP
using Colors
using StatsBase
using SumOfSquares
using LinearAlgebra
Random.seed!(1)
# Y = rand(side, side)
side = 10
num_pixels = side^2

function true_dens()
    p = rand(num_pixels)
    # sigma = rand(num_pixels, num_pixels)
    p[1] = min(mean(p[2:end]) + 0.2, 1.0)
    return p
end
# Gray.(mean(reshape(true_dens(), side, side) for _ in 1:100))

function invert_poly(poly, val, x)
    p(y) = poly(vcat(x, y))
    y_lower = 0.0
    y_upper = 1.0
    y_mid = 0.5
    while abs(y_lower - y_upper) > 1e-5
        p_lower = p(y_lower)
        p_upper = p(y_upper)
        p_mid = p(y_mid)
        if val > p_mid
            y_lower = y_mid
        else
            y_upper = y_mid
        end
        y_mid = (y_lower + y_upper) / 2
    end
    return y_mid
end

# affine coupling
# function main()
#     # X = [MNIST.traintensor(i) for i in 1:1] #[rand(side, side) for _ in 1:1000] # 60000 matrices 28x28
#     num_obs = 1000
#     X = [rand(side, side) for _ in 1:num_obs]
#
#     model = Model(() -> Hypatia.Optimizer(init_tol_qr = 1e-12))
#     polys = [] # Vector{JuMP.VariableRef}(undef, num_pixels - 1)
#     @polyvar x[1:(num_pixels - 1)]
#     for i in 1:(num_pixels - 1)
#         push!(polys, @variable(model, variable_type = Poly(monomials(x[1:i], 0:2))))
#     end
#     @variables(model, begin
#         log_epi
#         logdet_hypo
#         scalings[1:num_pixels] # == 1
#         consts[1:num_pixels]#  == 0
#     end)
#     log_likl = zeros(JuMP.AffExpr, num_obs * num_pixels)
#     idx = 1
#     for j in eachindex(X)
#         Xj = X[j]
#         log_likl[idx] = Xj[1] * scalings[1] + consts[1]
#         idx += 1
#         for i in 2:num_pixels
#             v = polys[i - 1](Xj[1:(i - 1)]) + Xj[i] * scalings[i] + consts[i]
#             log_likl[idx] = v
#             idx += 1
#         end
#     end
#     @constraint(model, vcat(logdet_hypo, 1, scalings) in Hypatia.HypoPerLogCone{Float64}(num_pixels + 2))
#     @constraint(model, vcat(-log_epi - num_pixels * num_obs * log(2π) / 2, 1, log_likl) in RotatedSecondOrderCone())
#     @objective(model, Max, log_epi + logdet_hypo * num_obs)
#     optimize!(model)
#     # @show JuMP.value.(polys)
#     # @show value.(scalings)
#
#     num_samples = 1000
#     x_out = [zeros(num_pixels) for _ in 1:num_samples]
#     for j in 1:num_samples
#         x_in = randn(num_pixels)
#         x_out[j][1] = (x_in[1] - value(consts[1])) / value(scalings[1])
#         for i in 2:num_pixels
#             x_out[j][i] = (x_in[i] - value(polys[i - 1](x_out[j][1:(i - 1)])) - value(consts[i])) / value(scalings[i])
#         end
#     end
#     x_out_mean = mean(x_out)
#     @show x_out_mean
#     Gray.(reshape(x_out_mean, side, side))
# end

# affine coupling without PolyJuMP
function main()
    # X = [MNIST.traintensor(i) for i in 1:1] #[rand(side, side) for _ in 1:1000] # 60000 matrices 28x28
    num_obs = 100
    X = [rand(side, side) for _ in 1:num_obs]
    X_flat = vcat([reshape(x, 1, num_pixels) for x in X]...)
    halfdeg = 1

    # model = Model(() -> Hypatia.Optimizer(init_tol_qr = 1e-12))
    model = Model(() -> Hypatia.Optimizer(
        use_dense_model = false,
        system_solver = Hypatia.Solvers.SymIndefSparseSystemSolver{Float64}(),
        )
        )
    polys = [] # Vector{JuMP.VariableRef}(undef, num_pixels - 1)
    @polyvar x[1:(num_pixels - 1)]
    for i in 1:(num_pixels - 1)
        push!(polys, @variable(model, [1:binomial(i + 2 * halfdeg, i)]))
    end
    @variables(model, begin
        log_epi
        logdet_hypo
        scalings[1:num_pixels] # == 1
        consts[1:num_pixels]#  == 0
    end)
    log_likl = zeros(JuMP.AffExpr, num_obs, num_pixels)
    idx = 1
    uni_cheb_polys = Matrix{Float64}[]
    log_likl[:, 1] = X_flat[:, 1] * scalings[1] .+ consts[1]
    for i in 2:num_pixels
        push!(uni_cheb_polys, calc_univariate_chebyshev(X_flat[:, i - 1], 2halfdeg))
        V = make_product_vandermonde(uni_cheb_polys, n_deg_exponents(i - 1, 2halfdeg))
        log_likl[:, i] = V * polys[i - 1] + X_flat[:, i] * scalings[i] .+ consts[i]
    end
    @constraint(model, vcat(logdet_hypo, 1, scalings) in Hypatia.HypoPerLogCone{Float64}(num_pixels + 2))
    @constraint(model, vcat(-log_epi - num_pixels * num_obs * log(2π) / 2, 1, vec(log_likl)) in RotatedSecondOrderCone())
    @objective(model, Max, log_epi + logdet_hypo * num_obs)
    @show "got model"
    optimize!(model)
    # @show JuMP.value.(polys)
    # @show value.(scalings)

    num_samples = 1000
    x_out = [zeros(num_pixels) for _ in 1:num_samples]
    for j in 1:num_samples
        x_in = randn(num_pixels)
        x_out[j][1] = (x_in[1] - value(consts[1])) / value(scalings[1])
        uni_cheb_polys = Matrix{Float64}[]
        for i in 2:num_pixels
            push!(uni_cheb_polys, calc_univariate_chebyshev([x_out[j][i - 1]], 2halfdeg))
            V = make_product_vandermonde(uni_cheb_polys, n_deg_exponents(i - 1, 2halfdeg))
            poly_val = (V * value.(polys[i - 1]))[1] # comes out as vector
            x_out[j][i] = (x_in[i] - poly_val - value(consts[i])) / value(scalings[i])
        end
    end
    x_out_mean = mean(x_out)
    @show x_out_mean
    Gray.(reshape(x_out_mean, side, side))
end

# monotonic SOS
# function main()
#     # X = [MNIST.traintensor(i) for i in 1:1] #[rand(side, side) for _ in 1:1000] # 60000 matrices 28x28
#     num_obs = 100
#     # X = [rand(side, side) for _ in 1:num_obs]
#     X = [reshape(true_dens(), side, side) for _ in 1:num_obs]
#
#     model = SOSModel(Hypatia.Optimizer)
#     polys = []
#     @polyvar x[1:num_pixels]
#     for i in 1:num_pixels
#         push!(polys, @variable(model, variable_type = Poly(monomials(x[1:i], 0:2))))
#         # push!(polys, @variable(model, [1:binomial(i + 2, i)]))
#     end
#     @variables(model, begin
#         log_epi
#         logdet_hypo
#     end)
#     diags = zeros(JuMP.AffExpr, num_pixels, num_obs)
#     log_likl = zeros(JuMP.AffExpr, length(X) * num_pixels)
#     idx = 1
#     for i in 1:num_pixels
#         # monos = monomials(x[1:i], 0:2)
#         # deriv = differentiate.(monos, x[i])
#         # @constraint(model, dot(polys[i], deriv) >= 0)
#         deriv = differentiate(polys[i], x[i])
#         @constraint(model, deriv >= 0)
#         for j in 1:num_obs
#             Xj = X[j]
#             diags[i, j] = deriv(Xj[1:i])
#             log_likl[idx] = polys[i](Xj[1:i])
#             # deriv_evals = [d(Xj[1:i]) for d in deriv]
#             # diags[i, j] = dot(deriv_evals, polys[i])
#             # mono_evals = [m(Xj[1:i]) for m in monos]
#             # log_likl[idx] = dot(mono_evals, polys[i])
#             idx += 1
#         end
#     end
#     @constraint(model, vcat(logdet_hypo, 1, vec(diags)) in Hypatia.HypoPerLogCone{Float64}(num_pixels * num_obs + 2))
#     @constraint(model, vcat(-log_epi - num_pixels * num_obs * log(2π) / 2, 1, log_likl) in RotatedSecondOrderCone())
#     @objective(model, Max, log_epi + logdet_hypo)
#     optimize!(model)
#     # @show JuMP.value.(polys)
#
#     num_samples = 10
#     x_out = [zeros(num_pixels) for _ in 1:num_samples]
#     for j in 1:num_samples
#         x_in = randn(num_pixels)
#         x_out[j][1] = invert_poly(value(polys[1]), x_in[1], Float64[])
#         for i in 2:num_pixels
#             x_out[j][i] = invert_poly(value(polys[i]), x_in[i], x_out[j][1:(i - 1)]) # TODO check in/out
#         end
#     end
#     x_out_mean = mean(x_out)
#     @show x_out_mean
#     Gray.(reshape(x_out_mean, side, side))
# end

# # monotonic SOS without SOS.jl
# function main()
#     # X = [MNIST.traintensor(i) for i in 1:1] #[rand(side, side) for _ in 1:1000] # 60000 matrices 28x28
#     num_obs = 100
#     X = [rand(side, side) for _ in 1:num_obs]
#     deg = 2
#     halfdeg = div(deg + 1, 2)
#     grad_halfdeg = div(deg, 2)
#
#     model = Model(Hypatia.Optimizer)
#
#     polys = []
#     Fs = []
#     for i in 1:num_pixels
#         dom = FreeDomain{Float64}(i)
#         (U, points, Ps, V) = interpolate(dom, halfdeg, calc_V = true) #, sample_factor = 2)
#         push!(Fs, qr!(Array(V'), Val(true)))
#         push!(polys, @variable(model, [1:U]))
#
#         (grad_U, grad_points, grad_Ps, _) = interpolate(dom, grad_halfdeg, calc_V = false) #, sample_factor = 2)
#         univ_chebs_derivs = [calc_univariate_chebyshev(grad_points[:, j], 2halfdeg, calc_gradient = true) for j in 1:i]
#         univ_chebs_g = [univ_chebs_derivs[j][(i == j) ? 2 : 1] for j in 1:i]
#         V_g = make_product_vandermonde(univ_chebs_g, n_deg_exponents(i, 2halfdeg))
#         # scal = inv(maximum(abs, V_g) / 10)
#         # scal < 1e-7 && @warn("model is numerically challenging to set up", maxlog = 1)
#         # lmul!(scal, V_g)
#         g_points_polys = Fs[i] \ V_g'
#         grad_interp = g_points_polys' * polys[i]
#         @constraint(model, grad_interp in Hypatia.WSOSInterpNonnegativeCone{Float64, Float64}(grad_U, grad_Ps))
#     end
#
#     @variables(model, begin
#         log_epi
#         logdet_hypo
#     end)
#
#     diags = zeros(JuMP.AffExpr, num_pixels, num_obs)
#     log_likl = zeros(JuMP.AffExpr, num_pixels, num_obs)
#     for j in 1:num_obs
#         Xj = X[j]
#         for i in 1:num_pixels
#             V_X = make_chebyshev_vandermonde(reshape(Xj[1:i], 1, i), 2halfdeg)
#             log_likl[i, j] = dot(polys[i], Fs[i] \ V_X')
#
#             univ_chebs_derivs = [calc_univariate_chebyshev([Xj[k]], 2halfdeg, calc_gradient = true) for k in 1:i]
#             univ_chebs_g = [univ_chebs_derivs[k][(i == k) ? 2 : 1] for k in 1:i]
#             V_g = make_product_vandermonde(univ_chebs_g, n_deg_exponents(i, 2halfdeg))
#             diags[i, j] = dot(polys[i], Fs[i] \ V_g')
#         end
#     end
#
#     @constraint(model, vcat(logdet_hypo, 1, vec(diags)) in Hypatia.HypoPerLogCone{Float64}(num_pixels * num_obs + 2))
#     @constraint(model, vcat(-log_epi - num_pixels * num_obs * log(2π) / 2, 1, vec(log_likl)) in RotatedSecondOrderCone())
#     @objective(model, Max, log_epi + logdet_hypo)
#     optimize!(model)
#     # @show JuMP.value.(polys)
#
#     function func_from_i(i)
#         function ret(x)
#             V_X = make_chebyshev_vandermonde(reshape(x[1:i], 1, i), 2halfdeg)
#             return dot(value.(polys[i]), Fs[i] \ V_X')
#         end
#         return ret
#     end
#
#     num_samples = 10
#     x_out = [zeros(num_pixels) for _ in 1:num_samples]
#     for j in 1:num_samples
#         x_in = randn(num_pixels)
#         x_out[j][1] = invert_poly(func_from_i(1), x_in[1], Float64[])
#         for i in 2:num_pixels
#             x_out[j][i] = invert_poly(func_from_i(i), x_in[i], x_out[j][1:(i - 1)]) # TODO check in/out
#         end
#     end
#     x_out_mean = mean(x_out)
#     @show x_out_mean
#     Gray.(reshape(x_out_mean, side, side))
# end

main()

# model = Model(Hypatia.Optimizer)
# d = 2
# U = 2 * d + 1
# (U, _, Ps, V, _) = ModelUtilities.interpolate(FreeDomain{Float64}(1), d, calc_V = true, calc_w = false)
# F = qr!(Array(V'), Val(true))
# V_X = ModelUtilities.make_chebyshev_vandermonde(X, 2 * d)
# X_pts_polys = F \ V_X'
# @variable(model, poly[1:U, num_pixels])
# @variable(model, z)
# @objective(model, Max, z)
# @constraint(model, vcat(z, X_pts_polys' * f_pts) in log cone)
