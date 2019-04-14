#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Data obtained from http://www.nber.org/data/nbprod2005.html
=#

using DataFrames
using CSV
using TimerOutputs
using SumOfSquares
using LinearAlgebra
# using Plots
include(joinpath(@__DIR__(), "jump.jl"))

# Example 1 from https://arxiv.org/pdf/1509.08165v1.pdf
function normfunction_data(; n::Int = 1, num_points::Int = 1000)
    f = x -> sum(abs2, x)
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 3.0)
    return (X, y, n)
end

# Example 5 from https://arxiv.org/pdf/1509.08165v1.pdf
function customfunction_data(; n::Int = 5, num_points::Int = 1000)
    f = x -> (5x[1] + 0.5x[2] + x[3])^2 + sqrt(x[4]^2 + x[5]^2)
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 9.0)
    return (X, y, n)
end

# Example 3 from https://arxiv.org/pdf/1509.08165v1.pdf
function production_data()
    df = CSV.read(joinpath(@__DIR__, "data", "naics5811.csv"))
    deleterows!(df, 157) # outlier
    # number of non production employees
    df[:prode] .= df[:emp] - df[:prode]
    # group by industry codes
    dfagg = aggregate(dropmissing(df), :naics, sum)
    # four covariates: non production employees, production worker hours, production workers, total capital stock
    n = 3
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


function make_model()
    return SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer,
        use_dense = true,
        verbose = true,
        system_solver = SO.QRCholCombinedHSDSystemSolver,
        linear_model = MO.PreprocessedLinearModel,
        max_iters = 1000,
        time_limit = 3.6e3,
        tol_rel_opt = 1e-5,
        tol_abs_opt = 1e-6,
        tol_feas = 1e-4,
        ))
end

# function run_hard_shapeconregr()
    reset_timer!(Hypatia.to)
    # degrees = 4:2:4
    d = 3; s = normfunction_data

    datasets = [
        # production_data,
        customfunction_data,
        # normfunction_data,
        ]

    # for d in degrees, s in datasets

        println()
        @show d
        @show s
        println()

        (X, y, n) = s(n = 5)
        dom = MU.Box(-ones(n), ones(n))

        shape_data = ShapeData(dom, dom, zeros(n), 0)

        model = make_model()
        (regressor, lagrange_polys) = build_shapeconregr_WSOS(model, X, y, d, shape_data, use_scalar = false, add_regularization = true)
        (val, runtime, bytes, gctime, memallocs) = @timed JuMP.optimize!(model)
        # regfun1(x) = JuMP.value(regressor)(x)
        # pl = plot(regfun1, -1, 1, lab = "No penalty")

        # x = collect(Float16, range(-1,length=100,stop=1));
        # y = collect(Float16, range(-1,length=100,stop=1));
        # z = [JuMP.value(regressor)([x[i]; y[i]]) for i in eachindex(x)]
        # pl = plot(x, y, z, st=:surface,camera=(-30,30))
        # scatter!(x, y, z)

        # func(x, y) = JuMP.value(regressor)([x; y])
        # p = plot(x, y, func, st = [:surface])
        # scatter3d!(X[:, 1], X[:, 2], y)


        # model = make_model()
        # (regressor, lagrange_polys) = build_shapeconregr_WSOS(model, X, y, d, shape_data, use_scalar = false, add_regularization = true)
        # (val, runtime, bytes, gctime, memallocs) = @timed JuMP.optimize!(model)
        # regfun2(x) = JuMP.value(regressor)(x)
        # plotlyjs()
        # plot!(pl, regfun2, -1, 1, lab = "With penalty")
        #
        # realfun(x) = exp(sum(abs2, x))
        # plot(pl, realfun, -1, 1, lab = "Truth")

        println()
        @show runtime
        @show bytes
        @show gctime
        @show memallocs
        println("\n\n")

        # pyplot()
        # x = collect(Float16, range(-2,length=100,stop=2));
        # y = collect(Float16, range(sqrt(2),length=100, stop=2));
        # z = (x.*y).-y.-x.+1;
        # surf(x,y,z);




    # end
# end

# run_hard_shapeconregr()



function makeplot(regressor, X, y)
    data_trace = scatter3d(
        x = X[:, 1],
        y = X[:, 2],
        z = y[:],
        mode = "markers",
        opacity = 0.8,
        marker_size = 6,
        marker_line_width = 0.5,
        marker_line_color = "rgba(217, 217, 217, 0.14)"
    )
    randx = rand(Distributions.Uniform(-1, 1), 200)
    randy = rand(Distributions.Uniform(-1, 1), 200)
    randz = [JuMP.value(regressor)([randx[i]; randy[i]]) for i in 1:200]
    # randz = [regressor(hcat(randx, randy)[i, :]) for i in 1:200]
    model_trace = mesh3d(
        x = randx,
        y = randy,
        z = randz,
        mode = "markers",
        opacity = 0.4,
        marker_size = 6,
        marker_line_width = 0.5,
        marker_line_color = "rgba(217, 217, 217, 0.14)"
    )
    layout = Layout(margin = attr(l = 0, r = 0, t = 0, b = 0))

    p = plot([data_trace, model_trace], layout)

    # 2d plot
    data_trace = scatter(
        x = X[:, 1],
        y = X[:, 2],
        z = y[:],
        mode = "markers",
        opacity = 0.8,
        marker_size = 6,
        marker_line_width = 0.5,
        marker_color = "orange",
        marker_line_color = "rgba(217, 217, 217, 0.14)"
    )
    x = collect(Float16, range(-1,length=100,stop=1));
    y = collect(Float16, range(-1,length=100,stop=1));
    z = [JuMP.value(regressor)([x[i]; y[i]]) for i in eachindex(x)]
    model_trace = contour(
        x = x,
        y = y,
        z = [JuMP.value(regressor)([x[i]; y[i]]) for i in 1:100],
    )
    p = plot([data_trace, model_trace], layout)
    # savefig(p, "psd_plot.pdf")

    return p
end
