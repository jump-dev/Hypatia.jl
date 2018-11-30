#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
using DataFrames
using MLDataUtils
using CSV
include(joinpath(@__DIR__(), "jump.jl"))
include(joinpath(@__DIR__(), "plot.jl"))

# Example 1 from Chapter 8 of thesis by G. Hall (2018).
function exprmnt1_data(;
    n::Int = 2,
    signal_ratio::Float64 = 0.0,
    )

    (l, u) = (0.5, 2.0)
    mono_domain = Hypatia.Box(l*ones(n), u*ones(n))
    conv_domain = Hypatia.Box(l*ones(n), u*ones(n))
    mono_profile = ones(n)
    conv_profile = 1.0
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)

    f = x -> exp(norm(x))
    npoints = 100
    (X, y) = generateregrdata(f, l, u, n, npoints, signal_ratio=signal_ratio)
    reference_rmse = sqrt(sum(abs2.([y[i] - f(X[i,:]) for i in 1:npoints])) / npoints)

    return (reference_rmse, X, y, shape_data)
end

function exprmnt_mdl(
    X,
    y,
    shapedata::ShapeData;
    deg::Int = 2,
    use_wsos::Bool = true,
    ignore_mono::Bool = false,
    ignore_conv::Bool = false,
    )

    if use_wsos
        tm = @elapsed begin
            (model, p) = build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=true, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
            JuMP.optimize!(model)
        end
    else
        tm = @elapsed begin
            (model, p) = build_shapeconregr_PSD(X, y, deg, shapedata, use_leastsqobj=true, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
            JuMP.optimize!(model)
        end
    end
    rmse = sqrt(JuMP.objective_value(model)) * sqrt(size(X, 1))
    return (rmse, tm, p)
end


# synthetic data
function runexp1()
    n = 4
    signal_options = [0.0, 100.0, 10.0, 2.0]
    shape_options = [true, false]
    wsos_options = [true, false]
    deg_options = 2:7
    outfilename = joinpath(@__DIR__(), "shapeconregr_$(round(Int, time()/10)).csv")

    open(outfilename, "w") do f
        println(f, "# n = $n")
        println(f, "signal_ratio,refrmse,deg,ignore_mono,ignore_conv,use_wsos,rmse,tm")
        # degrees of freedom for data
        for signal_ratio in signal_options
            (refrmse, X, y, shape_data) = exprmnt1_data(n=n, signal_ratio=signal_ratio)
            # degrees of freedom in the model
            for deg in deg_options, ignore_mono in shape_options, ignore_conv in shape_options, use_wsos in wsos_options
                println("running ", "signal_ratio=$signal_ratio, deg=$deg, ignore_mono=$ignore_mono, ignore_conv=$ignore_conv, use_wsos=$use_wsos")
                (rmse, tm, p) = exprmnt_mdl(X, y, shape_data, deg=deg, use_wsos=use_wsos, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
                println(f, "$signal_ratio,$refrmse,$deg,$ignore_mono,$ignore_conv,$use_wsos,$rmse,$tm")
                if runmakeplot
                    makeplot(p, X, y, joinpath(@__DIR__(), "plot$(signal_ratio)_$(deg)_$(ignore_mono)_$(ignore_conv)_$(use_wsos).pdf", l=(0.5, 0.5), u=(2.0, 2.0)))
                end
            end # model
        end # data
    end # do
end

# Example 3 from Chapter 8 of thesis by G. Hall (2018).
function exprmnt3_data()
    Random.seed!(1)
    df = CSV.read(joinpath(@__DIR__(), "wages/wages.csv"))
    # df = CSV.read("examples/shapeconregr/wages/wages.csv")
    inds = sample(1:size(df, 1), 2000, replace=false)
    X = convert(Array, df[inds, 2:3])
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

# real data
function runexp3()
    n = 2
    shape_options = [false]
    wsos_options = [true]
    deg_options = 2:2
    outfilename = joinpath(@__DIR__(), "shapeconregr_$(round(Int, time()/10)).csv")

    # open(outfilename, "w") do f
        # println(f, "fold,deg,ignore_mono,ignore_conv,use_wsos,train_rmse,test_rmse,test_rmse,tm")

            (folds, shape_data) = exprmnt3_data()
            foldcount = 0

            for ((Xtrain, ytrain), (Xtest, ytest)) in folds
                foldcount += 1

                # degrees of freedom in the model
                for deg in deg_options, ignore_mono in shape_options, ignore_conv in shape_options, use_wsos in wsos_options
                    Xtemp = convert(Array{Float64,2}, Xtrain') # TODO p(X) less strictly typed
                    (train_rmse, tm, p) = exprmnt_mdl(Xtemp, ytrain, shape_data, deg=deg, use_wsos=use_wsos, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
                    test_rmse = sum(abs2(ytest[i] - JuMP.value(p)(convert(Array{Float64,2}, Xtest)[:,i])) for i in 1:size(Xtest, 1))
                    # println(f, "$foldcount,$deg,$ignore_mono,$ignore_conv,$use_wsos,$train_rmse,$test_rmse,$tm")
                    if runmakeplot
                        p = makeplot(p, Xtemp, ytrain, filename=joinpath(@__DIR__(), "plot$(deg)_$(ignore_mono)_$(ignore_conv)_$(use_wsos).pdf"), l=(0.0, 0.0), u=(1.0, 1.0))
                        return p
                    end
                end # model

            end

    # end # do
end

runmakeplot = false
p = runexp3()

# n = 5, d = 6 , use convexity both struggle
# n = 4, d = 5, use both mosek
# small n high d to replicate papp numerical difficulties

# n = 5
# # degrees of freedom for data
# signal_ratio = 0.0 # 10.0
# (refrmse, X, y, shape_data) = exprmnt1_data(n=n, signal_ratio=signal_ratio)
# # degrees of freedom in the model
# deg = 5; ignore_mono = true; ignore_conv = false; use_wsos = true
# @time (rmse, tm, p) = exprmnt_mdl(X, y, shape_data, deg=deg, use_wsos=use_wsos, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
# p = makeplot(p, X, y, filename=joinpath(@__DIR__(), "mosek_both.pdf"), l=0.5, u=2.0)
