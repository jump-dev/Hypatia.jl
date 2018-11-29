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

function exprmnt1_mdl(
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
    n = 2
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
                (rmse, tm, p) = exprmnt1_mdl(X, y, shape_data, deg=deg, use_wsos=use_wsos, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
                println(f, "$signal_ratio,$refrmse,$deg,$ignore_mono,$ignore_conv,$use_wsos,$rmse,$tm")
                if runmakeplot
                    makeplot(p, X, y, joinpath(@__DIR__(), "plot$(signal_ratio)_$(deg)_$(ignore_mono)_$(ignore_conv)_$(use_wsos).pdf"))
                end
            end # model
        end # data
    end # do
end

# Example 3 from Chapter 8 of thesis by G. Hall (2018).
function exprmnt3_data()
    Random.seed!(seed)
    df = CSV.read(joinpath(@__DIR__(), "wages/wages.csv"))
    X = df[2:3]
    y = df[1]
    folds = kfolds((X, y); k = 10)

    # (big_X, big_y), (test_X, test_y) = splitobs(shuffleobs(X, y), at=0.75)
    # (train_X, train_y), (valid_X, valid_y) = splitobs(shuffleobs(big_X, big_y), at=0.67)

    education_interval = (l=0.0, u=18.0)
    experience_interval = (l=-4.0, u=63.0)
    mono_domain = Hypatia.Box([0.0, -4.0], [18.0, 63.0])
    conv_domain = Hypatia.Box([0.0, -4.0], [18.0, 63.0])
    mono_profile = ones(2)
    conv_profile = 1.0
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)

    return (folds, shape_data)
end

# real data
function runexp3()
    shape_options = [true, false]
    wsos_options = [true, false]
    deg_options = 2:2
    outfilename = joinpath(@__DIR__(), "shapeconregr_$(round(Int, time()/10)).csv")

    open(outfilename, "w") do f
        println(f, "fold,deg,ignore_mono,ignore_conv,use_wsos,train_rmse,test_rmse,test_rmse,tm")

            (folds, shape_data) = exprmnt3_data(n=n, signal_ratio=signal_ratio)

            for ((Xtrain, ytrain), (Xtest, ytest)) in folds

                # degrees of freedom in the model
                for deg in deg_options, ignore_mono in shape_options, ignore_conv in shape_options, use_wsos in wsos_options
                    (train_rmse, tm, p) = exprmnt1_mdl(Xtrain, ytrain, shape_data, deg=deg, use_wsos=use_wsos, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
                    test_rmse = sum(abs2([ytest[i] - p(Xtest[i,:]) for i in 1:size(Xtest, 1)]))
                    println(f, "$foldi,$deg,$ignore_mono,$ignore_conv,$use_wsos,$train_rmse,$test_rmse,$tm")
                    if runmakeplot
                        makeplot(p, Xtrain, ytrain, joinpath(@__DIR__(), "plot$(deg)_$(ignore_mono)_$(ignore_conv)_$(use_wsos).pdf"))
                    end
                end # model

            end

    end # do
end

# runmakeplot = false
# runexp1()
