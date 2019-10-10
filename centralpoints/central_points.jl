#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#

using Optim
using LineSearches
using DataFrames
using CSV
using ForwardDiff

function hypoperlogdettri_obj(x, n)
    (u, v, w) = (x[1], x[2], x[3])
    return -log(v * n * log(w / v) - u) - n * log(w) - log(v) + (abs2(u) + abs2(v) + abs2(w) * n) / 2
end

function epiperexp_obj(x, n)
    (u, v, w) = (x[1], x[2], x[3])
    if log(u / v) - log(n * exp(w / v)) > 0
        return -log(log(u / v) - log(n * exp(w / v))) - log(u) - 2 * log(v) + (abs2(u) + abs2(v) + abs2(w) * n) / 2
    else
        # sometimes we step too far
        return Inf
    end
end

function hypoperlog_obj(x, n)
    (u, v, w) = (x[1], x[2], x[3])
    return -log(v * n * (log(w / v)) - u) - n * log(w) - log(v) + (abs2(u) + abs2(v) + abs2(w) * n) / 2
end

function hypogeomean_obj(x, alpha)
    (u, w) = (x[1], x[2:end])
    return -log(prod((w[i] / alpha[i])^alpha[i] for i in 1:length(alpha)) + u) - sum((1 .- alpha) .* log.(w ./ alpha)) - log(-u) + sum(abs2, w) / 2 + abs2(u) / 2
end

barrier_dict = Dict(
    :hypoperlogdettri => (hypoperlogdettri_obj, n -> [-1.0, 1.0, 1.0], [-Inf, 0.0, 0.0], [Inf, Inf, Inf]),
    :epiperexp => (epiperexp_obj, n -> [2.0, 1.0, -log(n)], [0.0, 0.0, -Inf], [Inf, Inf, Inf]),
    :hypoperlog => (hypoperlog_obj, n -> [-1.0, 1.0, 1.0], [-Inf, 0.0, 0.0], [Inf, Inf, Inf]),
    )

function get_central_pt(cone_name)
    (obj, initial_point, lower, upper) = barrier_dict[cone_name]
    res_df = DataFrame(n = Int[], u = Float64[], v = Float64[], w = Float64[])
    for n in unique(round.(Int, exp.(0:0.1:12)))
        initial_x = initial_point(n)
        f(x) = obj(x, n)
        grad(x) = ForwardDiff.gradient(f, x)
        hess(x) = ForwardDiff.hessian(f, x)
        df = TwiceDifferentiable(f, grad, hess, initial_x, inplace = false)
        dfc = TwiceDifferentiableConstraints(lower, upper)
        res = optimize(df, dfc, initial_x, IPNewton())
        x = Optim.minimizer(res)
        push!(res_df, [n, x...])
    end
    return res_df
end

hypoperlogdettri_df = get_central_pt(:hypoperlogdettri)
CSV.write(joinpath(@__DIR__(), "hypoperlogdettri.csv"), hypoperlogdettri_df)
epiperexp_df = get_central_pt(:epiperexp)
CSV.write(joinpath(@__DIR__(), "epiperexp.csv"), epiperexp_df)
hypoperlog_df = get_central_pt(:hypoperlog)
CSV.write(joinpath(@__DIR__(), "hypoperlog.csv"), hypoperlog_df)

function get_central_pt_geomean()
    obj = hypogeomean_obj
    ns = Int[]
    alphas = Float64[]
    us = Float64[]
    ws = Float64[]
    nsamples = 20
    for n in [1, 2, 5, 8, 10, 20, 40, 40, 50]
        initial_x = ones(n + 1)
        (lower, upper) = ([-Inf, fill(0.0, n)...], fill(Inf, n + 1))
        for sample in 1:nsamples
            # sample uniformly from the simplex
            alpha = log.(rand(n))
            alpha ./= sum(alpha)
            initial_x[1] = -prod(alpha[i] ^ (-alpha[i]) for i in eachindex(alpha)) / (n + 1)
            f(x) = obj(x, alpha)
            grad(x) = ForwardDiff.gradient(f, x)
            hess(x) = ForwardDiff.hessian(f, x)
            df = TwiceDifferentiable(f, grad, hess, initial_x, inplace = false)
            dfc = TwiceDifferentiableConstraints(lower, upper)
            res = optimize(df, dfc, initial_x, IPNewton())
            x = Optim.minimizer(res)
            for i in 1:n
                push!(ns, n)
                push!(alphas, alpha[i])
                push!(us, x[1])
                push!(ws, x[1 + i])
            end
        end
    end
    res_df = DataFrame(n = ns, alpha = alphas, u = us, w = ws)
    return res_df
end

hypogeomean_df = get_central_pt_geomean()
CSV.write(joinpath(@__DIR__(), "hypogeomean.csv"), hypogeomean_df)
