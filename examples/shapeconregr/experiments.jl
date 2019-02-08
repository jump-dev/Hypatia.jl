#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
using DataFrames
using MLDataUtils
using CSV
include(joinpath(@__DIR__(), "jump.jl"))
include(joinpath(@__DIR__(), "plot.jl"))

# Example 1 from Chapter 8 of thesis by G. Hall (2018).
function syn1_data(;
    n::Int = 2,
    signal_ratio::Float64 = 0.0,
    )

    (l, u) = (0.5, 2.0)
    mono_domain = Hypatia.Ball(1.25*ones(n), 1.25*sqrt(n))
    conv_domain = Hypatia.Ball(1.25*ones(n), 1.25*sqrt(n))
    mono_profile = ones(Int, n)
    conv_profile = 1
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)

    f = x -> exp(norm(x))
    num_points = 100
    (X, y) = generateregrdata(f, l, u, n, num_points, signal_ratio=signal_ratio)
    reference_rmse = sqrt(sum(abs2.([y[i] - f(X[i,:]) for i in 1:num_points])) / num_points)

    folds = kfolds((X', y); k = 5)

    return (reference_rmse, folds, shape_data)
end

# Example function from Papp thesis
function syn2_data(;
    n::Int = 2,
    signal_ratio::Float64 = 0.0,
    )

    (l, u) = (0.0, 1.0)
    mono_domain = Hypatia.Ball(ones(n), sqrt(n))
    conv_domain = Hypatia.Ball(ones(n), sqrt(n))
    mono_profile = ones(Int, n)
    conv_profile = -1
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)

    f = x -> inv(1 + exp(-10.0 * norm(x)))
    num_points = 100
    (X, y) = generateregrdata(f, l, u, n, num_points, signal_ratio=signal_ratio)
    reference_rmse = sqrt(sum(abs2.([y[i] - f(X[i,:]) for i in 1:num_points])) / num_points)

    folds = kfolds((X', y); k = 5)

    return (reference_rmse, folds, shape_data)
end

# Example 3 from Chapter 8 of thesis by G. Hall (2018).
function real1_data()
    Random.seed!(1)
    df = CSV.read(joinpath(@__DIR__(), "wages/wages.csv"))
    # df = CSV.read("examples/shapeconregr/wages/wages.csv")
    inds = sample(1:size(df, 1), 2000, replace=false)
    X = convert(Matrix, df[inds, 2:3])
    y = convert(Array, df[inds, 1])
    # normalize
    X = (X .- minimum(X, dims=1)) ./ (maximum(X, dims=1) - minimum(X, dims=1))
    y = (y .- minimum(y)) ./  (maximum(y, dims=1) - minimum(y, dims=1))
    println("making folds")
    folds = kfolds((X', y); k = 3)

    # (big_X, big_y), (test_X, test_y) = splitobs(shuffleobs(X, y), at=0.75)
    # (train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs(big_X, big_y), at=0.67)

    education_interval = (l=0.0, u=18.0)
    experience_interval = (l=-4.0, u=63.0)
    mono_domain = Hypatia.Box([0.0, -4.0], [18.0, 63.0])
    conv_domain = Hypatia.Box([0.0, -4.0], [18.0, 63.0])
    mono_profile = [1.0, 0.0]
    conv_profile = -1.0
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)

    return (folds, shape_data)
end

# Example 3 from https://arxiv.org/pdf/1509.08165v1.pdf
function real2_data()
    df = CSV.read(joinpath(dirname(dirname(Base.find_package("Hypatia"))), "data", "naics5811.csv"))
    df[:prode] .= df[:emp] - df[:prode]
    dfagg = aggregate(dropmissing(df), :naics, sum) # TODO ask what Rahul did with missing values
    X = convert(Matrix, dfagg[[:prode_sum, :prodh_sum, :prodw_sum, :cap_sum]])
    y = convert(Array, dfagg[:vship_sum])
    Xlog = log.(X)
    # normalize
    Xlog .-= sum(Xlog, dims = 1) / size(Xlog, 1)
    y .-= sum(y) / length(y)
    Xlog ./= sqrt.(sum(abs2, Xlog, dims = 1))
    y ./= sqrt(sum(abs2, y))

    println("making folds")
    folds = kfolds((X', y); k = 5)

    mono_domain = Hypatia.Box(-ones(4), ones(4))
    conv_domain = Hypatia.Box(-ones(4), ones(4))
    mono_profile = zeros(4)
    conv_profile = 1.0
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)

    return (folds, shape_data)
end

# Example 1 from https://arxiv.org/pdf/1509.08165v1.pdf
function syn3_data(;
    num_points::Int = 500,
    n::Int = 2,
    )
    mono_domain = Hypatia.FreeDomain(n)
    conv_domain = Hypatia.FreeDomain(n)
    mono_profile = zeros(n)
    conv_profile = 1
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)

    f = x -> sum(abs2, x)
    (X, y) = generateregrdata(f, -1.0, 1.0, n, num_points, signal_ratio = 3.0)
    reference_rmse = sqrt(sum(abs2.([y[i] - f(X[i,:]) for i in 1:num_points])) / num_points)

    folds = kfolds((X', y); k = 5)

    return (reference_rmse, folds, shape_data)
end

# Example 2 from https://arxiv.org/pdf/1509.08165v1.pdf
function syn4_data()
    syn3_data(num_points = 1000, n = 10)
end

# Example 5 from https://arxiv.org/pdf/1509.08165v1.pdf
function syn5_data(;
    num_points::Int = 1000,
    n::Int = 5,
    )
    mono_domain = Hypatia.FreeDomain(n)
    conv_domain = Hypatia.FreeDomain(n)
    mono_profile = zeros(n)
    conv_profile = 1
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)

    f = x -> (5x[1] + 0.5x[2] + x[3])^2 + sqrt(x[4]^2 + x[5]^2)
    (X, y) = generateregrdata(f, -1.0, 1.0, n, num_points, signal_ratio = 3.0)
    reference_rmse = sqrt(sum(abs2.([y[i] - f(X[i,:]) for i in 1:num_points])) / num_points)

    folds = kfolds((X', y); k = 5)

    return (reference_rmse, folds, shape_data)
end

# synthetic data
function runexp_syn(i::Int)
    for n in 1:6
        signal_options = [10.0]
        mono_options = [false]
        conv_options = [true]
        wsos_options = [true]
        deg_options = 3:7
        outfilename = joinpath(@__DIR__(), "shapeconregr_$(round(Int, time()/10)).csv")

        open(outfilename, "w") do f
            println(f, "# n = $n, sample_pts = $sample_pts, use_leastsqobj = $use_leastsqobj, testfunc = $i")
            println(f, "fold,signal_ratio,refrmse,deg,use_wsos,tr_rmse,ts_rmse,tm,b_tm,s_tm,status")
            # degrees of freedom for data
            for signal_ratio in signal_options
                if i == 1
                    (refrmse, folds, shape_data) = syn1_data(n=n, signal_ratio=signal_ratio)
                elseif i == 2
                    (refrmse, folds, shape_data) = syn2_data(n=n, signal_ratio=signal_ratio)
                elseif i == 3
                    (refrmse, folds, shape_data) = syn3_data()
                elseif i == 4
                    (refrmse, folds, shape_data) = syn4_data()
                elseif i == 5
                    (refrmse, folds, shape_data) = syn5_data()
                else
                    error()
                end
                foldcount = 0

                for ((Xtrain, ytrain), (Xtest, ytest)) in folds
                    foldcount += 1
                    Xtemp = convert(Array{Float64,2}, Xtrain')

                    # degrees of freedom in the model
                    for deg in deg_options, use_wsos in wsos_options
                        println("running ", "signal_ratio=$signal_ratio, deg=$deg, use_wsos=$use_wsos")
                        (mdl, tr_rmse, tm, b_tm, s_tm, regr) = exprmnt_mdl(Xtemp, ytrain, shape_data, deg=deg, use_wsos=use_wsos)
                        ts_rmse = sum(abs2(ytest[i] - JuMP.value(regr)(convert(Array{Float64,2}, Xtest)[:,i])) for i in 1:size(Xtest, 1))
                        println(f, "$foldcount,$signal_ratio,$refrmse,$deg,$use_wsos,$tr_rmse,$ts_rmse,$tm,$b_tm,$s_tm,$(JuMP.termination_status(mdl))")
                        if runmakeplot
                            makeplot(regr, Xtemp, ytrain, joinpath(@__DIR__(), "plot$(signal_ratio)_$(deg)_$(use_wsos).pdf", l=(0.5, 0.5), u=(2.0, 2.0)))
                        end
                    end # model
                end # folds
            end # data
        end # do
    end # n
end

# real data
function runexp_real()
    shape_options = [false]
    wsos_options = [true]
    deg_options = 3:3
    outfilename = joinpath(@__DIR__(), "shapeconregr_$(round(Int, time()/10)).csv")

    # open(outfilename, "w") do f
        # println(f, "fold,deg,use_wsos,train_rmse,test_rmse,test_rmse,tm")

            (folds, shape_data) = real1_data()
            foldcount = 0

            for ((Xtrain, ytrain), (Xtest, ytest)) in folds
                foldcount += 1

                # degrees of freedom in the model
                for deg in deg_options, use_wsos in wsos_options
                    Xtemp = convert(Array{Float64,2}, Xtrain') # TODO p(X) less strictly typed
                    (mdl, train_rmse, tm, regr) = exprmnt_mdl(Xtemp, ytrain, shape_data, deg=deg, use_wsos=use_wsos)
                    test_rmse = sum(abs2(ytest[i] - JuMP.value(regr)(convert(Array{Float64,2}, Xtest)[:,i])) for i in 1:size(Xtest, 1))
                    # println(f, "$foldcount,$deg,$use_wsos,$train_rmse,$test_rmse,$tm")
                    if runmakeplot
                        p = makeplot(regr, Xtemp[:, 1:2], ytrain, filename=joinpath(@__DIR__(), "plot$(deg)_$(use_wsos).pdf"), l=(0.0, 0.0), u=(1.0, 1.0))
                        return p
                    end
                end # model

            end

    # end # do
end

function exprmnt_mdl(
    X,
    y,
    shapedata::ShapeData;
    deg::Int = 3,
    use_wsos::Bool = true,
    )

    if use_wsos
        tm = @elapsed begin
            build_tm = @elapsed begin
                (model, p) = build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=use_leastsqobj, sample=sample_pts)
            end
            solve_tm = @elapsed JuMP.optimize!(model)
        end
    else
        tm = @elapsed begin
            build_tm = @elapsed begin
                (model, p) = build_shapeconregr_PSD(X, y, deg, shapedata, use_leastsqobj=use_leastsqobj)
            end
            solve_tm = @elapsed JuMP.optimize!(model)
        end
    end
    rmse = sqrt(JuMP.objective_value(model)) * sqrt(size(X, 1))
    return (model, rmse, tm, build_tm, solve_tm, p)
end

runmakeplot = false
sample_pts = true
use_leastsqobj = true
p = runexp_syn(5)
 # p = runexp_real()

# # quick runs:
# n = 2
# signal_ratio = 10.0
# (refrmse, folds, shape_data) = exprmnt1_data(n=n, signal_ratio=signal_ratio)
# ((Xtrain, ytrain), (_, _)) = folds[1]
# # degrees of freedom in the model
# deg = 5; use_wsos = true
# @time (mdl, rmse, tm, b_tm, s_tm, p) = exprmnt_mdl(convert(Array{Float64,2}, Xtrain'), ytrain, shape_data, deg=deg, use_wsos=use_wsos)
# filename = "" # joinpath(@__DIR__(), "mosek_both.pdf")
# pl = makeplot(p, convert(Array{Float64,2}, Xtrain'), ytrain, filename=filename, l=(0.5,0.5), u=(2.0,2.0))
