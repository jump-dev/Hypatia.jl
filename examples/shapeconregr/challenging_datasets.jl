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
    return (X, y, n, f)
end

function expfunction_data(; n::Int = 5, num_points::Int = 100)
    f = x -> exp(sum(abs2, x))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 9.0)
    return (X, y, n, f)
end

# Example 5 from https://arxiv.org/pdf/1509.08165v1.pdf
function customfunction_data(; n::Int = 5, num_points::Int = 100)
    f = x -> (5x[1] + 0.5x[2] + x[3])^2 + sqrt(x[4]^2 + x[5]^2)
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 9.0)
    return (X, y, n, f)
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

function build_solve_model(X, y, shapedata, deg, use_wsos)

    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer,
        use_dense = true,
        verbose = true,
        system_solver = SO.QRCholCombinedHSDSystemSolver,
        linear_model = MO.PreprocessedLinearModel,
        max_iters = 750,
        time_limit = 3.6e3,
        tol_rel_opt = 1e-5,
        tol_abs_opt = 1e-6,
        tol_feas = 1e-6,
        ))

    if use_wsos
        p = build_shapeconregr_WSOS(model, X, y, deg, shapedata)
        solve_tm = @elapsed JuMP.optimize!(model)
    else
        p = build_shapeconregr_PSD(model, X, y, deg, shapedata)
        solve_tm = @elapsed JuMP.optimize!(model)
    end
    rmse = sqrt(max(JuMP.objective_value(model), 0)) * sqrt(size(X, 1))
    return (model, rmse, solve_tm, p)
end

function run_synthetic()
    deg_options = [3; 4; 5]
    wsos_options = [false]
    conv_options = [true, false]
    for n in [4]
        outfilename = joinpath(@__DIR__(), "shapeconregr_$(round(Int, time() / 10)).csv")
        (X, y, _, func) = expfunction_data(n = n)
        folds = kfolds((X', y), k = 5)
        open(outfilename, "w") do f
            println(f, "# n = $n")
            println(f, "fold,refrmse,deg,use_wsos,tr_rmse,ts_rmse,time,conv,status")
            foldcount = 0

            for ((Xtrain, ytrain), (Xtest, ytest)) in folds
                foldcount += 1
                Xtemp = convert(Array{Float64,2}, Xtrain')
                num_points = size(Xtemp, 1)
                refrmse = sqrt(sum(abs2.([ytrain[i] - func(Xtemp[i,:]) for i in 1:num_points])) / num_points)

                # degrees of freedom in the model
                for deg in deg_options, use_wsos in wsos_options, conv in conv_options
                    if conv == false
                        shape_data = ShapeData(MU.Box(-ones(n), ones(n)), MU.Box(-ones(n), ones(n)), ones(n), 0)
                    else
                        shape_data = ShapeData(MU.Box(-ones(n), ones(n)), MU.Box(-ones(n), ones(n)), zeros(n), 1)
                    end
                    println("running ", "deg = $deg, use_wsos = $use_wsos, n = $n, conv = $conv")
                    (mdl, tr_rmse, s_tm, regr) = build_solve_model(Xtemp, ytrain, shape_data, deg, use_wsos)
                    ts_rmse = sum(abs2(ytest[i] - JuMP.value(regr)(convert(Array{Float64,2}, Xtest)[:,i])) for i in 1:size(Xtest, 1))
                    println(f, "$foldcount,$refrmse,$deg,$use_wsos,$tr_rmse,$ts_rmse,$s_tm,$conv,$(JuMP.termination_status(mdl))")
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

        (X, y, n) = s()
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
