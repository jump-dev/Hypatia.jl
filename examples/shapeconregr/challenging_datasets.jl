#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

# Data obtained from http://www.nber.org/data/nbprod2005.html
=#

using DataFrames
using CSV
include(joinpath(@__DIR__(), "jump.jl"))

# Example 1 from https://arxiv.org/pdf/1509.08165v1.pdf
function normfunction_data(num_points::Int = 500)
    n = 5
    f = x -> sum(abs2, x)
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 9.0)
    return (X, y, n)
end

# Example 5 from https://arxiv.org/pdf/1509.08165v1.pdf
function customfunction_data(num_points::Int = 500)
    n = 5
    f = x -> (5x[1] + 0.5x[2] + x[3])^2 + sqrt(x[4]^2 + x[5]^2)
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = 9.0)
    return (X, y, n)
end

# Example 3 from https://arxiv.org/pdf/1509.08165v1.pdf
function production_data()
    df = CSV.read(joinpath(@__DIR__, "data", "naics5811.csv"))
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
    return (X, y, n)
end

function build_model(dataset::Int, deg::Int)
    if dataset == 1
        (X, y, n) = normfunction_data()
    elseif dataset == 2
        (X, y, n) = customfunction_data()
    elseif dataset == 3
        (X, y, n) = production_data()
    else
        error()
    end
    mono_domain = MU.FreeDomain(n)
    conv_domain = MU.FreeDomain(n)
    mono_profile = zeros(n)
    conv_profile = 1
    shape_data = ShapeData(mono_domain, conv_domain, mono_profile, conv_profile)
    return build_shapeconregr_WSOS(X, y, deg, shape_data)
end

(model, poly) = build_model(1, 4)
JuMP.optimize!(model)
