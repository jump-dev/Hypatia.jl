#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#
include(joinpath(@__DIR__(), "jump.jl"))

# Example 1 from Chapter 8 of thesis by G. Hall (2018).
function exprmnt1_data(;
    n::Int = 2,
    signal_ratio::Float64 = 0.0,
    )

    (l, u) = (0.5, 2.0)
    domain = Hypatia.Box(l*ones(n), u*ones(n))
    mono_profile = ones(n)
    conv_profile = 1.0
    shape_data = ShapeData(domain, domain, mono_profile, conv_profile)

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
    rmse = sqrt(JuMP.ojbective_value(model)) * sqrt(npoints)
    return (rmse, tm)
end


function runall()
    n = 2
    signal_options = [0.0, 100.0, 10.0, 2.0]
    shape_options = [true, false]
    wsos_options = [true, false]
    deg_options = 2:7
    outfilename = joinpath(@__DIR__(), "shapeconregr_$(round(Int, time()/10))")

    open(outfilename, "w") do f
        println(f, "# n = $n")
        println(f, "signal_ratio,refrmse,d,ignore_mono,ignore_conv,use_wsos,rmse,tm")
        # degrees of freedom for data
        for signal_ratio in signal_options
            (refrmse, X, y, shape_data) = exprmnt1_data(n=n, signal_ratio=signal_ratio)
            # degrees of freedom in the model
            for deg in deg_options, ignore_mono in shape_options, ignore_conv in shape_options, use_wsos in wsos_options
                (rmse, tm) = exprmnt1_mdl(X, y, shape_data, deg=deg, use_wsos=use_wsos, ignore_mono=ignore_mono, ignore_conv=ignore_conv)
                println(f, "$signal_ratio,$refrmse,$d,$ignore_mono,$ignore_conv,$use_wsos,$rmse,$tm")
            end # model
        end # data
    end # do
end

runall()
