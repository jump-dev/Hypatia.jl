#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
==#

using PlotlyJS
import Distributions

function densityest_plot(
    regressor::DynamicPolynomials.Polynomial{true,Float64},
    X::Matrix{Float64};
    random_data::Bool = false,
    use_contour::Bool = true
    )
    (num_obs, dim) = size(X)
    if random_data
        x = rand(Distributions.Uniform(-1, 1), 5000, dim)
        z = [regressor(x[i, :]) for i in 1:5000]
    else
        x = X[:, 1:2]
        z = [regressor(X[i, :]) for i in 1:num_obs]
    end
    if use_contour
        model_trace = contour(
            x = x[:, 1],
            y = x[:, 2],
            z = z,
        )
    else
        model_trace = mesh3d(
            x = x[:, 1],
            y = x[:, 2],
            z = z,
            mode = "markers",
            opacity = 0.4,
            marker_size = 6,
            marker_line_width = 0.5,
            marker_line_color = "rgba(217, 217, 217, 0.14)",
        )
    end
    data_trace = scatter(
        x = X[:, 1],
        y = X[:, 2],
        mode = "markers",
        opacity = 0.8,
        color = "black",
        marker_size = 8,
        marker_line_width = 0.5,
        marker_line_color = "rgba(217, 217, 217, 0.14)",
        # marker_line_color = "rgba(0, 0, 0, 0.4)",
    )
    layout = Layout(
        margin = attr(l = 0, r = 0, t = 0, b = 0),
        title = "Density for sepal length and width",
        xlabel = "not working",
        )
    return plot([model_trace, data_trace], layout)
end
