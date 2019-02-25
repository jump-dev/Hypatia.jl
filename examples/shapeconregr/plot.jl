#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
==#

using PlotlyJS
# using ORCA
# using Plots
# plotlyjs()
import Distributions

function get_data_trace3d(X, y)
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
    return data_trace
end

function get_model_trace3d(regressor, n)
    s = 20
    a = collect(range(-1, stop = 1, length = s))
    x = zeros(s^n, n)
    q = 0
    for i in 1:s, j in 1:s
        q += 1
        x[q, 1] = a[i]
        x[q, 2] = a[j]
    end
    z = [regressor(x[i, :]) for i in 1:s^2]
    model_trace = mesh3d(
        x = x[:, 1],
        y = x[:, 2],
        z = z,
        mode = "markers",
        opacity = 0.4,
        marker_size = 6,
        marker_line_width = 0.5,
        marker_line_color = "rgba(217, 217, 217, 0.14)"
    )
    return model_trace
end

# generic plot for synthetic data
function makeplot3d(regressor, X, y)
    data_trace = get_data_trace3d(X, y)
    model_trace = get_model_trace3d(regressor, size(X, 2))
    layout = Layout(margin = attr(l = 0, r = 0, t = 0, b = 0))

    pl = plot([data_trace, model_trace], layout)

    return pl
end

function get_data_trace2d(X, y)
    data_trace = scatter(
        x = X[:, 1],
        y = y,
        mode = "markers",
        opacity = 0.8,
        marker_size = 6,
        marker_line_width = 0.5,
        marker_line_color = "rgba(217, 217, 217, 0.14)"
    )
    return data_trace
end

function get_model_trace2d(regressor, X)
    perm = sortperm(X[:, 1])
    X = X[perm, :]
    y = [regressor(X[i, :]) for i in 1:size(X, 1)]
    model_trace = scatter(
        x = X[:, 1],
        y = y,
        mode = "markers + lines",
        opacity = 0.4,
        marker_size = 6,
        marker_line_width = 0.5,
        marker_line_color = "rgba(217, 217, 217, 0.14)"
    )
    return model_trace
end

function makeplot2d(regressor, X, y)
    data_trace = get_data_trace2d(X, y)
    layout = Layout(margin = attr(l = 0, r = 0, t = 0, b = 0), title = "Value of Shipment")

    pl = plot([data_trace], layout)

    return pl
end

# plot for real dataset
function makeplot_production()
    (X, y) = production_data()
    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # X = X[:, 1:1]
    deg = 3
    n = size(X, 2)
    shapedata = ShapeData(MU.FreeDomain(n), MU.FreeDomain(n), zeros(n), 1)
    poly = build_shapeconregr_WSOS(model, X[:, 1:n], y, deg, shapedata, use_naive = false)
    JuMP.optimize!(model)
    pl = makeplot2d(JuMP.value(poly), X, y)
end

function makeplot_exp()
    (X, y) = expfunction_data(n = 2, num_points = 200)
    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    deg = 6
    n = size(X, 2)
    shapedata = ShapeData(MU.FreeDomain(n), MU.FreeDomain(n), ones(n), 0)
    poly = build_shapeconregr_WSOS(model, X[:, 1:n], y, deg, shapedata)
    JuMP.optimize!(model)
    pl = makeplot3d(JuMP.value(poly), X, y)
end
