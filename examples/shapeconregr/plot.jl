#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
==#

using JuMP
using PolyJuMP
using MultivariatePolynomials
using DynamicPolynomials
using PlotlyJS
using ORCA
using Distributions
using Test

function makeplot(regressor,
    X,
    y;
    filename::String = "plot.pdf",
    l::Tuple{Float64,Float64} = (-1.0, -1.0),
    u::Tuple{Float64,Float64} = (1.0, 1.0),
    )
    data_trace = scatter3d(
        x=X[:, 1],
        y=X[:, 2],
        z=y[:],
        mode="markers",
        opacity=0.8,
        marker_size=6,
        marker_line_width=0.0,
        marker_line_color="rgba(217, 217, 217, 0.14)"
    )
    randx = rand(Uniform(l[1], u[1]), 200)
    randy = rand(Uniform(l[2], u[2]), 200)
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
    savefig(p, filename)

    return p
end
