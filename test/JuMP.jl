#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

function solveandcheck_JuMP(model, truemin; atol=1e-4, rtol=1e-4)
    JuMP.optimize!(model)
    term_status = JuMP.termination_status(model)
    primal_obj = JuMP.objective_value(model)
    dual_obj = JuMP.objective_bound(model)
    pr_status = JuMP.primal_status(model)
    du_status = JuMP.dual_status(model)
    @test term_status == MOI.OPTIMAL
    @test pr_status == MOI.FEASIBLE_POINT
    @test du_status == MOI.FEASIBLE_POINT
    @test primal_obj ≈ dual_obj atol=atol rtol=rtol
    @test primal_obj ≈ truemin atol=atol rtol=rtol
end

function namedpoly1_JuMP()
    # the Heart polynomial in a box
    (x, f, dom, truemin) = getpolydata(:heart)
    # WSOS formulation
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d=2)
    solveandcheck_JuMP(model, truemin)
end

function namedpoly2_JuMP()
    # the Schwefel polynomial in a box
    (x, f, dom, truemin) = getpolydata(:schwefel)
    # WSOS formulation
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d=2)
    solveandcheck_JuMP(model, truemin)
    # SDP formulation
    # model = build_JuMP_namedpoly_PSD(x, f, dom, d=2)
    # solveandcheck_JuMP(model, truemin)
end

function namedpoly3_JuMP()
    # the Magnetism polynomial in a ball
    (x, f, dom, truemin) = getpolydata(:magnetism7_ball)
    # WSOS formulation
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d=2)
    solveandcheck_JuMP(model, truemin)
    # SDP formulation
    # model = build_JuMP_namedpoly_PSD(x, f, dom, d=2)
    # solveandcheck_JuMP(model, truemin)
end

function namedpoly4_JuMP()
    # the Motzkin polynomial in an ellipsoid containing two local minima in opposite orthants
    (x, f, dom, truemin) = getpolydata(:motzkin_ellipsoid)
    # WSOS formulation
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d=4)
    solveandcheck_JuMP(model, truemin)
    # SDP formulation
    # model = build_JuMP_namedpoly_PSD(x, f, dom, d=4)
    # solveandcheck_JuMP(model, truemin)
end

function namedpoly5_JuMP()
    (x, f, dom, truemin) = getpolydata(:caprasse)
    # WSOS formulation
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d=4)
    solveandcheck_JuMP(model, truemin)
    # SDP formulation
    # model = build_JuMP_namedpoly_PSD(x, f, dom, d=4)
    # solveandcheck_JuMP(model, truemin)
end

function namedpoly6_JuMP()
    (x, f, dom, truemin) = getpolydata(:goldsteinprice)
    # WSOS formulation
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d=7)
    solveandcheck_JuMP(model, truemin)
    # SDP formulation
    # model = build_JuMP_namedpoly_PSD(x, f, dom, d=7)
    # solveandcheck_JuMP(model, truemin)
end

function namedpoly7_JuMP()
    (x, f, dom, truemin) = getpolydata(:lotkavolterra)
    # WSOS formulation
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d=3)
    solveandcheck_JuMP(model, truemin)
    # SDP formulation
    # model = build_JuMP_namedpoly_PSD(x, f, dom, d=3)
    # solveandcheck_JuMP(model, truemin)
end

function namedpoly8_JuMP()
    (x, f, dom, truemin) = getpolydata(:robinson)
    # WSOS formulation
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d=8)
    solveandcheck_JuMP(model, truemin)
    # SDP formulation
    # model = build_JuMP_namedpoly_PSD(x, f, dom, d=8)
    # solveandcheck_JuMP(model, truemin)
end

function namedpoly9_JuMP()
    (x, f, dom, truemin) = getpolydata(:reactiondiffusion_ball)
    # WSOS formulation
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d=3)
    solveandcheck_JuMP(model, truemin)
    # SDP formulation
    # model = build_JuMP_namedpoly_PSD(x, f, dom, d=3)
    # solveandcheck_JuMP(model, truemin)
end

function namedpoly10_JuMP()
    (x, f, dom, truemin) = getpolydata(:rosenbrock)
    # WSOS formulation
    model = build_JuMP_namedpoly_WSOS(x, f, dom, d=5)
    solveandcheck_JuMP(model, truemin, atol=1e-3)
    # SDP formulation
    # model = build_JuMP_namedpoly_PSD(x, f, dom, d=5)
    # solveandcheck_JuMP(model, truemin, atol=1e-3)
end

function shapeconregr1_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 0.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 4.4065e-1
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr2_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^3))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 1.3971e-1
    solveandcheck_JuMP(model, truemin)
    # test with non-sampling based interpolation
    (model, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false, sample=false)
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr3_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^4))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 2.4577e-1
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr4_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^3))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 1.5449e-1
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr5_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^4))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 2.5200e-1
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr6_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 0.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=true)
    truemin = 5.4584e-2
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr7_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^4))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=true)
    truemin = 3.3249e-2
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr8_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 4, 100, 0.0, x -> -inv(1 + exp(-10.0 * norm(x))))
    (X, y) = generateregrdata(f, 0.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    shapedata = ShapeData(MU.Box(zeros(n), ones(n)), MU.Box(zeros(n), ones(n)), ones(n), 1)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=true)
    truemin = 3.7723e-03
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr9_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 4, 100, 10.0, x -> -inv(1 + exp(-10.0 * norm(x))))
    (X, y) = generateregrdata(f, 0.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    shapedata = ShapeData(MU.Box(zeros(n), ones(n)), MU.Box(zeros(n), ones(n)), ones(n), 1)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=true)
    truemin = 3.0995e-02 # not verified with SDP
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr10_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 4, 100, 0.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=true)
    truemin = 5.0209e-02 # not verified with SDP
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr11_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 5, 100, 10.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, 0.5, 2.0, n, npoints, signal_ratio=signal_ratio)
    shapedata = ShapeData(MU.Box(0.5*ones(n), 2*ones(n)), MU.Box(0.5*ones(n), 2*ones(n)), ones(n), 1)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, shapedata, use_leastsqobj=true)
    truemin = 0.22206 # not verified with SDP
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr12_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 5, 100, 10.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, 0.5, 2.0, n, npoints, signal_ratio=signal_ratio)
    shapedata = ShapeData(MU.Box(0.5*ones(n), 2*ones(n)), MU.Box(0.5*ones(n), 2*ones(n)), ones(n), 1)
    (model, p) = build_shapeconregr_PSD(X, y, deg, shapedata, use_leastsqobj=true)
    truemin = 0.22206 # not verified with SDP
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr13_JuMP()
    (n, deg, npoints, signal_ratio, f) = (2, 6, 100, 1.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_WSOS(X, y, deg, ShapeData(n), use_leastsqobj=false)
    truemin = 1.7751 # not verified with SDP
    solveandcheck_JuMP(model, truemin)
end

function shapeconregr14_JuMP() # out of memory error when converting sparse to dense in MOI conversion
    (n, deg, npoints, signal_ratio, f) = (5, 5, 1000, 0.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_PSD(X, y, deg, ShapeData(n), use_leastsqobj=true, dense=true)
    JuMP.optimize!(model)
end

function shapeconregr15_JuMP() # out of memory error during preprocessing
    (n, deg, npoints, signal_ratio, f) = (5, 5, 1000, 0.0, x -> exp(norm(x)))
    (X, y) = generateregrdata(f, -1.0, 1.0, n, npoints, signal_ratio=signal_ratio)
    (model, p) = build_shapeconregr_PSD(X, y, deg, ShapeData(n), use_leastsqobj=true, dense=false)
    JuMP.optimize!(model)
end
