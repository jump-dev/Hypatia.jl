#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
==#

using PlotlyJS
# using ORCA
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
    s = 5
    a = collect(range(-1, stop = 1, length = s))
    x = zeros(s^n, n)
    q = 0
    for i in 1:s, j in 1:s, k in 1:s, l in 1:s
        q += 1
        x[q, 1] = a[i]
        x[q, 2] = a[j]
        x[q, 3] = a[k]
        x[q, 4] = a[l]
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
    model_trace = get_model_trace2d(regressor, X)
    layout = Layout(margin = attr(l = 0, r = 0, t = 0, b = 0))

    pl = plot([data_trace, model_trace], layout)

    return pl
end

# plot for real dataset
function makeplot_production()
    (X, y, _) = production_data(n = 2)
    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    deg = 4
    n = size(X, 2)
    shapedata = ShapeData(MU.FreeDomain(n), MU.FreeDomain(n), zeros(n), 1)
    poly = build_shapeconregr_WSOS(model, X[:, 1:n], y, deg, shapedata, use_naive = true)
    JuMP.optimize!(model)
    pl = makeplot2d(JuMP.value(poly), X, y)
end

# 8.281059404049783e-6x₁³ + 6.882751952330422e-5x₁²x₂ + 0.00013183157318747492x₁x₂² + 6.461202451160328e-5x₂³ - 0.04670076489051557
