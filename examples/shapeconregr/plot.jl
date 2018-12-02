#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
==#

using JuMP
using PolyJuMP
using MultivariatePolynomials
using DynamicPolynomials
using PlotlyJS
using ORCA
using Test

function makeplot(regressor, X, y)
    data_trace = scatter3d(
        x=X[:, 1],
        y=X[:, 2],
        z=y[:],
        mode="markers",
        opacity=0.8,
        marker_size=6,
        marker_line_width=0.5,
        marker_line_color="rgba(217, 217, 217, 0.14)"
    )
    randx = rand(Uniform(-1, 1), 200)
    randy = rand(Uniform(-1, 1), 200)
    randz = [JuMP.value(regressor)(hcat(randx, randy)[i,:]) for i in 1:200]
    mdl_trace = mesh3d(
        x=randx,
        y=randy,
        z=randz,
        mode="markers",
        opacity=0.4,
        marker_size=6,
        marker_line_width=0.5,
        marker_line_color="rgba(217, 217, 217, 0.14)"
    )
    layout = Layout(margin=attr(l=0, r=0, t=0, b=0))

    p = plot([data_trace, mdl_trace], layout)
    # savefig(p, "psd_plot.pdf")

    return p
end
