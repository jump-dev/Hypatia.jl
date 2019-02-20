#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

Data obtained from http://www.nber.org/data/nbprod2005.html
=#

using DataFrames
using CSV
using MLDataUtils
include(joinpath(@__DIR__(), "jump.jl"))

# Example 1 from https://arxiv.org/pdf/1509.08165v1.pdf
function normfunction_data(; n::Int = 5, num_points::Int = 100)
    f = x -> sum(abs2, x)
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 9.0)
    return (X, y, f)
end

function expfunction_data(; n::Int = 5, num_points::Int = 100)
    f = x -> exp(sum(abs2, x))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 9.0)
    return (X, y, f)
end

# Example 5 from https://arxiv.org/pdf/1509.08165v1.pdf
function customfunction_data(; n::Int = 5, num_points::Int = 100)
    @assert n = 5
    f = x -> (5x[1] + 0.5x[2] + x[3])^2 + sqrt(x[4]^2 + x[5]^2)
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 9.0)
    return (X, y, f)
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
    return (Xlog, y)
end

function build_solve_model(X, y, shapedata, deg, use_wsos)

    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer,
        use_dense = true,
        verbose = true,
        system_solver = SO.QRCholCombinedHSDSystemSolver,
        linear_model = MO.PreprocessedLinearModel,
        max_iters = 750,
        time_limit = 300.0,
        tol_rel_opt = 1e-5,
        tol_abs_opt = 1e-6,
        tol_feas = 1e-6,
        ))

    if use_wsos
        poly = build_shapeconregr_WSOS(model, X, y, deg, shapedata)
        (val, runtime, bytes, gctime, memallocs) = try_to_solve(model)
    else
        poly = build_shapeconregr_PSD(model, X, y, deg, shapedata)
        (val, runtime, bytes, gctime, memallocs) = try_to_solve(model)
    end
    if isfinite(runtime)
        rmse = sqrt(max(JuMP.objective_value(model), 0)) * sqrt(size(X, 1))
    else
        rmse = Inf
    end
    return (model, rmse, runtime, poly)
end

function rmse(X, y, func)
    num_points = size(X, 1)
    return sqrt(sum(abs2.([y[i] - func(X[i,:]) for i in 1:num_points])) / num_points)
end

function try_to_solve(model)
    try
        (val, runtime, bytes, gctime, memallocs) = @timed JuMP.optimize!(model)
        return (val, runtime, bytes, gctime, memallocs)
    catch
        return (Inf, Inf, Inf, Inf, Inf)
    end
end

function run_experiments(issynthetic = true)
    deg_options = [3]
    wsos_options = [true, false]
    conv_options = ["mono", "conv", "neither"]
    nrange = [3]
    for n in nrange # data options
        outfilename = joinpath(@__DIR__(), "shapecon_$(round(Int, time() / 10)).csv")
        if issynthetic
            (X, y, func) = expfunction_data(n = n)
        else
            (X, y) = production_data()
        end
        folds = kfolds((X', y), k = 5)
        open(outfilename, "a") do f
            println(f, "# n = $n")
            println(f, "fold,tr_refrmse,ts_refrmse,deg,use_wsos,tr_rmse,ts_rmse,time,conv,status")
            foldcount = 0

            for ((Xtrain, ytrain), (Xtest, ytest)) in folds
                foldcount += 1
                tr_Xarr = convert(Array{Float64,2}, Xtrain')
                ts_Xarr = convert(Array{Float64,2}, Xtest')
                if issynthetic
                    tr_refrmse = rmse(tr_Xarr, ytrain, func)
                    ts_refrmse = rmse(ts_Xarr, ytest, func)
                else
                    tr_refrmse = 0
                    ts_refrmse = 0
                end

                for deg in deg_options, use_wsos in wsos_options, conv in conv_options # model options
                    if conv == "mono"
                        shape_data = ShapeData(MU.Box(-ones(n), ones(n)), MU.Box(-ones(n), ones(n)), ones(Int, n), 0)
                    elseif conv == "conv"
                        shape_data = ShapeData(MU.Box(-ones(n), ones(n)), MU.Box(-ones(n), ones(n)), zeros(Int, n), 1)
                    else
                        shape_data = ShapeData(MU.Box(-ones(n), ones(n)), MU.Box(-ones(n), ones(n)), zeros(Int, n), 0)
                    end
                    println("running ", "deg = $deg, use_wsos = $use_wsos, n = $n, conv = $conv")
                    (mdl, tr_rmse, s_tm, regr) = build_solve_model(tr_Xarr, ytrain, shape_data, deg, use_wsos)
                    if isfinite(s_tm)
                        ts_rmse = rmse(ts_Xarr, ytest, JuMP.value(regr))
                    else
                        ts_rmse = Inf
                    end
                    println(f, "$foldcount,$tr_refrmse,$ts_refrmse,$deg,$use_wsos,$tr_rmse,$ts_rmse,$s_tm,$conv,$(JuMP.termination_status(mdl))")
                end # model
            end # folds
        end # do
    end # n
end

function run_hard_shapeconregr()
    degrees = 4:2:6

    datasets = [
        normfunction_data,
        customfunction_data,
        production_data,
        ]

    for d in degrees, s in datasets
        model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer,
            use_dense = true,
            verbose = true,
            system_solver = SO.QRCholCombinedHSDSystemSolver,
            linear_model = MO.PreprocessedLinearModel,
            max_iters = 250,
            time_limit = 3.6e3,
            tol_rel_opt = 1e-5,
            tol_abs_opt = 1e-6,
            tol_feas = 1e-6,
            ))

        println()
        @show d
        @show s
        println()

        (X, y) = s()
        n = size(X, 2)
        shape_data = ShapeData(MU.FreeDomain(n), MU.FreeDomain(n), zeros(n), 1)
        build_shapeconregr_WSOS(model, X, y, d, shape_data)

        (val, runtime, bytes, gctime, memallocs) = @timed JuMP.optimize!(model)

        println()
        @show runtime
        @show bytes
        @show gctime
        @show memallocs
        println("\n\n")
    end
end

# run_hard_shapeconregr()
