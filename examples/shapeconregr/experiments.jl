#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
using DataFrames
using MLDataUtils
using CSV
using TimerOutputs
include(joinpath(@__DIR__(), "jump.jl"))
include(joinpath(@__DIR__(), "plot.jl"))

# Example 1 from Chapter 8 of thesis by G. Hall (2018).
function exprmnt1_data(;
    n::Int = 2,
    signal_ratio::Float64 = 0.0,
    )

    (l, u) = (0.5, 2.0)
    # mono_domain = Hypatia.Box(l*ones(n), u*ones(n))
    mono_domain = Hypatia.Ball(zeros(n), sqrt(n))
    # conv_domain = Hypatia.Box(l*ones(n), u*ones(n))
    conv_domain = Hypatia.Ball(zeros(n), sqrt(n))
    mono_profile = ones(Int, n)
    conv_profile = 1
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)

    f = x -> exp(norm(x))
    npoints = 100
    (X, y) = generateregrdata(f, l, u, n, npoints, signal_ratio=signal_ratio)
    reference_rmse = sqrt(sum(abs2.([y[i] - f(X[i,:]) for i in 1:npoints])) / npoints)

    return (reference_rmse, X, y, shape_data)
end

# example function from Papp thesis
function exprmnt2_data(;
    n::Int = 2,
    signal_ratio::Float64 = 0.0,
    )

    (l, u) = (0.0, 1.0)
    # mono_domain = Hypatia.Box(l*ones(n), u*ones(n))
    mono_domain = Hypatia.Ball(ones(n), sqrt(n))
    # conv_domain = Hypatia.Box(l*ones(n), u*ones(n))
    conv_domain = Hypatia.Ball(ones(n), sqrt(n))
    mono_profile = ones(Int, n)
    conv_profile = -1
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)

    f = x -> inv(1 + exp(-10.0 * norm(x)))
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

    reset_timer!(Hypatia.to)

    if use_wsos
        tm = @elapsed begin
            (model, p) =
                build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=use_leastsqobj, ignore_mono=ignore_mono, ignore_conv=ignore_conv, sample_pts=sample_pts)
            JuMP.optimize!(model)
        end
    else
        tm = @elapsed begin
            (model, p) = build_shapeconregr_PSD(X, y, deg, shapedata, use_leastsqobj=use_leastsqobj, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
            JuMP.optimize!(model)
        end
    end
    rmse = sqrt(JuMP.objective_value(model)) * sqrt(size(X, 1))
    return (model, rmse, tm, p)
end


# synthetic data
function runexp12(i::Int)
    for n in 1:6
        signal_options = [0.0; 100.0; 400.0; 900.0]
        shape_options = [false]
        wsos_options = [true, false]
        deg_options = 2:6
        outfilename = joinpath(@__DIR__(), "shapeconregr_$(round(Int, time()/10)).csv")

        open(outfilename, "w") do f
            println(f, "# n = $n, sample_pts = $sample_pts, use_leastsqobj = $use_leastsqobj, testfunc = $i")
            println(f, "signal_ratio,refrmse,deg,ignore_mono,ignore_conv,use_wsos,rmse,tm,status")
            # degrees of freedom for data
            for signal_ratio in signal_options
                if i == 1
                    (refrmse, X, y, shape_data) = exprmnt1_data(n=n, signal_ratio=signal_ratio)
                elseif i == 2
                    (refrmse, X, y, shape_data) = exprmnt2_data(n=n, signal_ratio=signal_ratio)
                else
                    error()
                end
                # degrees of freedom in the model
                for deg in deg_options, ignore_mono in shape_options, ignore_conv in shape_options, use_wsos in wsos_options
                    ignore_conv = true
                    println("running ", "signal_ratio=$signal_ratio, deg=$deg, ignore_mono=$ignore_mono, ignore_conv=$ignore_conv, use_wsos=$use_wsos")
                    (mdl, rmse, tm, p) = exprmnt_mdl(X, y, shape_data, deg=deg, use_wsos=use_wsos, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
                    println(f, "$signal_ratio,$refrmse,$deg,$ignore_mono,$ignore_conv,$use_wsos,$rmse,$tm,$(JuMP.termination_status(mdl))")
                    if runmakeplot
                        makeplot(p, X, y, joinpath(@__DIR__(), "plot$(signal_ratio)_$(deg)_$(ignore_mono)_$(ignore_conv)_$(use_wsos).pdf", l=(0.5, 0.5), u=(2.0, 2.0)))
                    end
                end # model
            end # data
        end # do
    end # n
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
                    (mdl, train_rmse, tm, p) = exprmnt_mdl(Xtemp, ytrain, shape_data, deg=deg, use_wsos=use_wsos, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
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
sample_pts = true
use_leastsqobj = true
# p = runexp12(1)

# n = 5, d = 6 , use convexity both struggle
# n = 4, d = 5, use both mosek
# small n high d to replicate papp numerical difficulties
# exprmnt2 works n=2,deg=4,domain0->1, any objtype, sampling method, no noise
# go up to deg 5, noise 400 still ok
# with no noise, deg 6 ok
# exp2 with n=5, deg=5 worked with ls

n = 3
# degrees of freedom for data
signal_ratio = 400.0
(refrmse, X, y, shape_data) = exprmnt2_data(n=n, signal_ratio=signal_ratio)
# degrees of freedom in the model
deg = 4; ignore_mono = false; ignore_conv = false; use_wsos = true
@time (mdl, rmse, tm, p) = exprmnt_mdl(X, y, shape_data, deg=deg, use_wsos=use_wsos, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
filename = "" # joinpath(@__DIR__(), "mosek_both.pdf")
# pl = makeplot(p, X, y, filename=filename, l=(0.0,0.0), u=(2.0,2.0))
