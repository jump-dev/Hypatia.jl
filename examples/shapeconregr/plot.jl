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

function makeplot(sdp_p, wsos_p)
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
    sdpz = [JuMP.value(sdp_p)(hcat(randx, randy)[i,:]) for i in 1:200]
    wsosz = [JuMP.value(wsos_p)(hcat(randx, randy)[i,:]) for i in 1:200]
    sdp_trace = mesh3d(
        x=randx,
        y=randy,
        z=sdpz,
        mode="markers",
        opacity=0.4,
        marker_size=6,
        marker_line_width=0.5,
        marker_line_color="rgba(217, 217, 217, 0.14)"
    )
    wsos_trace = mesh3d(
        x=randx,
        y=randy,
        z=wsosz,
        mode="markers",
        opacity=0.4,
        marker_size=6,
        marker_line_width=0.5,
        marker_line_color="rgba(217, 217, 217, 0.14)"
    )
    layout = Layout(margin=attr(l=0, r=0, t=0, b=0))

    sdp_plot = plot([data_trace, sdp_trace], layout)
    wsos_plot = plot([data_trace, wsos_trace], layout)
    # savefig(sdp_plot, "sdp_plot.pdf")
    # savefig(wsos_plot, "wsos_plot.pdf")

end
