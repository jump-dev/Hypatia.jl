#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Data obtained from http://www.nber.org/data/nbprod2005.html
=#

using DataFrames
using CSV
using TimerOutputs
using SumOfSquares
using LinearAlgebra
using Plots
plotlyjs()
include(joinpath(@__DIR__(), "jump.jl"))

# Example 1 from https://arxiv.org/pdf/1509.08165v1.pdf
function normfunction_data(; n::Int = 1, num_points::Int = 1000)
    f = x -> exp(sum(abs2, x))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 9.0)
    return (X, y, n)
end

# Example 5 from https://arxiv.org/pdf/1509.08165v1.pdf
function customfunction_data(; n::Int = 5, num_points::Int = 100)
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
    d = 6; s = normfunction_data

    datasets = [
        # production_data,
        # customfunction_data,
        normfunction_data,
        ]

    # for d in degrees, s in datasets

        println()
        @show d
        @show s
        println()

        (X, y, n) = s(n = 4)
        dom = MU.Box(-ones(n), ones(n))

        shape_data = ShapeData(dom, dom, zeros(n), 0)

        model = make_model()
        (regressor, lagrange_polys) = build_shapeconregr_WSOS(model, X, y, d, shape_data, use_scalar = false, add_regularization = true)
        (val, runtime, bytes, gctime, memallocs) = @timed JuMP.optimize!(model)
        # regfun1(x) = JuMP.value(regressor)(x)
        # pl = plot(regfun1, -1, 1, lab = "No penalty")

        # model = make_model()
        # (regressor, lagrange_polys) = build_shapeconregr_WSOS(model, X, y, d, shape_data, use_scalar = false, add_regularization = true)
        # (val, runtime, bytes, gctime, memallocs) = @timed JuMP.optimize!(model)
        # regfun2(x) = JuMP.value(regressor)(x)
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




    # end
# end

# run_hard_shapeconregr()
