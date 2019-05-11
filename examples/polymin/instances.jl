#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

# JuMP examples, real

function JuMP_polymin1(; use_dense = false)
    # the Heart polynomial in a box
    (x, f, dom, true_obj) = getpolydata(:heart)
    return build_JuMP_polymin_WSOS(x, f, dom, d = 2)
end

function JuMP_polymin2(; use_dense = false)
    # the Schwefel polynomial in a box
    (x, f, dom, true_obj) = getpolydata(:schwefel)
    return build_JuMP_polymin_WSOS(x, f, dom, d = 2)
end

function JuMP_polymin3(; use_dense = false)
    # the Magnetism polynomial in a ball
    (x, f, dom, true_obj) = getpolydata(:magnetism7_ball)
    return build_JuMP_polymin_WSOS(x, f, dom, d = 2)
end

function JuMP_polymin4(; use_dense = false)
    # the Motzkin polynomial in an ellipsoid containing two local minima in opposite orthants
    (x, f, dom, true_obj) = getpolydata(:motzkin_ellipsoid)
    return build_JuMP_polymin_WSOS(x, f, dom, d = 4)
end

function JuMP_polymin5(; use_dense = false)
    (x, f, dom, true_obj) = getpolydata(:caprasse)
    return build_JuMP_polymin_WSOS(x, f, dom, d = 4)
end

function JuMP_polymin6(; use_dense = false)
    (x, f, dom, true_obj) = getpolydata(:goldsteinprice)
    return build_JuMP_polymin_WSOS(x, f, dom, d = 7)
end

function JuMP_polymin7(; use_dense = false)
    (x, f, dom, true_obj) = getpolydata(:lotkavolterra)
    return build_JuMP_polymin_WSOS(x, f, dom, d = 3)
end

function JuMP_polymin8(; use_dense = false)
    (x, f, dom, true_obj) = getpolydata(:robinson)
    return build_JuMP_polymin_WSOS(x, f, dom, d = 8)
end

function JuMP_polymin9(; use_dense = false)
    (x, f, dom, true_obj) = getpolydata(:reactiondiffusion_ball)
    return build_JuMP_polymin_WSOS(x, f, dom, d = 3)
end

function JuMP_polymin10(; use_dense = false)
    (x, f, dom, true_obj) = getpolydata(:rosenbrock)
    return = build_JuMP_polymin_WSOS(x, f, dom, d = 5)
end

# native examples, complex

function complexpolymin1(; primal_wsos = true)
    (n, deg, f, gs, gdegs, truemin) = complexpolys[:abs1d]
    d = 1
    return build_complexpolymin(n, d, f, gs, gdegs, primal_wsos)
end

function complexpolymin2(; primal_wsos = true)
    (n, deg, f, gs, gdegs, truemin) = complexpolys[:absunit1d]
    d = 1
    return build_complexpolymin(n, d, f, gs, gdegs, primal_wsos)
end

function complexpolymin3(; primal_wsos = true)
    (n, deg, f, gs, gdegs, truemin) = complexpolys[:negabsunit1d]
    d = 2
    return build_complexpolymin(n, d, f, gs, gdegs, primal_wsos)
end

function complexpolymin4(; primal_wsos = true)
    (n, deg, f, gs, gdegs, truemin) = complexpolys[:absball2d]
    d = 1
    return build_complexpolymin(n, d, f, gs, gdegs, primal_wsos)
end

function complexpolymin5(; primal_wsos = true)
    (n, deg, f, gs, gdegs, truemin) = complexpolys[:absbox2d]
    d = 2
    return build_complexpolymin(n, d, f, gs, gdegs, primal_wsos)
end

function complexpolymin6(; primal_wsos = true)
    (n, deg, f, gs, gdegs, truemin) = complexpolys[:negabsbox2d]
    d = 1
    return build_complexpolymin(n, d, f, gs, gdegs, primal_wsos)
end

function complexpolymin7(; primal_wsos = true)
    (n, deg, f, gs, gdegs, truemin) = complexpolys[:denseunit1d]
    d = 2
    return build_complexpolymin(n, d, f, gs, gdegs, primal_wsos)
end
