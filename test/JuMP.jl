#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

function solve_and_check_JuMP(model, true_obj; atol = 1e-3, rtol = 1e-3)
    JuMP.optimize!(model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
    @test JuMP.primal_status(model) == MOI.FEASIBLE_POINT
    @test JuMP.dual_status(model) == MOI.FEASIBLE_POINT
    @test JuMP.objective_value(model) ≈ JuMP.objective_bound(model) atol = atol rtol = rtol
    @test JuMP.objective_value(model) ≈ true_obj atol = atol rtol = rtol
end

function test_JuMP_polymin1()
    # the Heart polynomial in a box
    model = JuMP_polymin1()
    solve_and_check_JuMP(model, true_obj)
end

function polymin2_JuMP()
    # the Schwefel polynomial in a box
    (x, f, dom, true_obj) = getpolydata(:schwefel)
    # WSOS formulation
    model = build_JuMP_polymin_WSOS(x, f, dom, d = 2)
    solve_and_check_JuMP(model, true_obj)
    # SDP formulation
    # model = build_JuMP_polymin_PSD(x, f, dom, d = 2)
    # solve_and_check_JuMP(model, true_obj)
end

function polymin3_JuMP()
    # the Magnetism polynomial in a ball
    (x, f, dom, true_obj) = getpolydata(:magnetism7_ball)
    # WSOS formulation
    model = build_JuMP_polymin_WSOS(x, f, dom, d = 2)
    solve_and_check_JuMP(model, true_obj)
    # SDP formulation
    # model = build_JuMP_polymin_PSD(x, f, dom, d = 2)
    # solve_and_check_JuMP(model, true_obj)
end

function polymin4_JuMP()
    # the Motzkin polynomial in an ellipsoid containing two local minima in opposite orthants
    (x, f, dom, true_obj) = getpolydata(:motzkin_ellipsoid)
    # WSOS formulation
    model = build_JuMP_polymin_WSOS(x, f, dom, d = 4)
    solve_and_check_JuMP(model, true_obj)
    # SDP formulation
    # model = build_JuMP_polymin_PSD(x, f, dom, d = 4)
    # solve_and_check_JuMP(model, true_obj)
end

function polymin5_JuMP()
    (x, f, dom, true_obj) = getpolydata(:caprasse)
    # WSOS formulation
    model = build_JuMP_polymin_WSOS(x, f, dom, d = 4)
    solve_and_check_JuMP(model, true_obj)
    # SDP formulation
    # model = build_JuMP_polymin_PSD(x, f, dom, d = 4)
    # solve_and_check_JuMP(model, true_obj)
end

function polymin6_JuMP()
    (x, f, dom, true_obj) = getpolydata(:goldsteinprice)
    # WSOS formulation
    model = build_JuMP_polymin_WSOS(x, f, dom, d = 7)
    solve_and_check_JuMP(model, true_obj)
    # SDP formulation
    # model = build_JuMP_polymin_PSD(x, f, dom, d = 7)
    # solve_and_check_JuMP(model, true_obj)
end

function polymin7_JuMP()
    (x, f, dom, true_obj) = getpolydata(:lotkavolterra)
    # WSOS formulation
    model = build_JuMP_polymin_WSOS(x, f, dom, d = 3)
    solve_and_check_JuMP(model, true_obj)
    # SDP formulation
    # model = build_JuMP_polymin_PSD(x, f, dom, d = 3)
    # solve_and_check_JuMP(model, true_obj)
end

function polymin8_JuMP()
    (x, f, dom, true_obj) = getpolydata(:robinson)
    # WSOS formulation
    model = build_JuMP_polymin_WSOS(x, f, dom, d = 8)
    solve_and_check_JuMP(model, true_obj)
    # SDP formulation
    # model = build_JuMP_polymin_PSD(x, f, dom, d = 8)
    # solve_and_check_JuMP(model, true_obj)
end

function polymin9_JuMP()
    (x, f, dom, true_obj) = getpolydata(:reactiondiffusion_ball)
    # WSOS formulation
    model = build_JuMP_polymin_WSOS(x, f, dom, d = 3)
    solve_and_check_JuMP(model, true_obj)
    # SDP formulation
    # model = build_JuMP_polymin_PSD(x, f, dom, d = 3)
    # solve_and_check_JuMP(model, true_obj)
end

function polymin10_JuMP()
    (x, f, dom, true_obj) = getpolydata(:rosenbrock)
    # WSOS formulation
    model = build_JuMP_polymin_WSOS(x, f, dom, d = 5)
    solve_and_check_JuMP(model, true_obj, atol = 1e-3)
    # SDP formulation
    # model = build_JuMP_polymin_PSD(x, f, dom, d = 5)
    # solve_and_check_JuMP(model, true_obj, atol = 1e-3)
end

function test_JuMP_shapeconregr1()
    # (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> exp(norm(x)))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false)
    true_obj = 4.4065e-1
    model = JuMP_shapeconregr1()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr2_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^3))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false)
    true_obj = 1.3971e-1
    model = JuMP_shapeconregr2(sample = true)
    solve_and_check_JuMP(model, true_obj)
    model = JuMP_shapeconregr2(sample = false)
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr3_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^4))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false)
    model = JuMP_shapeconregr3()
    true_obj = 2.4577e-1
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr4_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^3))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false)
    model = JuMP_shapeconregr4()
    true_obj = 1.5449e-1
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr5_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^4))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false)
    model = JuMP_shapeconregr5()
    true_obj = 2.5200e-1
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr6_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> exp(norm(x)))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = true)
    model = JuMP_shapeconregr6()
    true_obj = 5.4584e-2
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr7_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^4))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = true)
    model = JuMP_shapeconregr7()
    true_obj = 3.3249e-2
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr8_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 4, 100, 0.0, x -> -inv(1 + exp(-10.0 * norm(x))))
    # (X, y) = generate_regr_data(f, 0.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # shape_data = ShapeData(MU.Box(zeros(n), ones(n)), MU.Box(zeros(n), ones(n)), ones(n), 1)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, shape_data, use_lsq_obj = true)
    model = JuMP_shapeconregr8()
    true_obj = 3.7723e-03
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr9_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 4, 100, 10.0, x -> -inv(1 + exp(-10.0 * norm(x))))
    # (X, y) = generate_regr_data(f, 0.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # shape_data = ShapeData(MU.Box(zeros(n), ones(n)), MU.Box(zeros(n), ones(n)), ones(n), 1)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, shape_data, use_lsq_obj = true)
    model = JuMP_shapeconregr9()
    true_obj = 3.0995e-02 # not verified with SDP
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr10_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 4, 100, 0.0, x -> exp(norm(x)))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = true)
    model = JuMP_shapeconregr10()
    true_obj = 5.0209e-02 # not verified with SDP
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr11_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 5, 100, 10.0, x -> exp(norm(x)))
    # (X, y) = generate_regr_data(f, 0.5, 2.0, n, num_points, signal_ratio = signal_ratio)
    # shape_data = ShapeData(MU.Box(0.5 * ones(n), 2 * ones(n)), MU.Box(0.5 * ones(n), 2 * ones(n)), ones(n), 1)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, shape_data, use_lsq_obj = true)
    model = JuMP_shapeconregr11()
    true_obj = 0.22206 # not verified with SDP
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr12_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 5, 100, 10.0, x -> exp(norm(x)))
    # (X, y) = generate_regr_data(f, 0.5, 2.0, n, num_points, signal_ratio = signal_ratio)
    # shape_data = ShapeData(MU.Box(0.5 * ones(n), 2 * ones(n)), MU.Box(0.5 * ones(n), 2 * ones(n)), ones(n), 1)
    # model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_PSD(model, X, y, deg, shape_data, use_lsq_obj = true)
    model = JuMP_shapeconregr12()
    true_obj = 0.22206 # not verified with SDP
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr13_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 6, 100, 1.0, x -> exp(norm(x)))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false)
    model = JuMP_shapeconregr13()
    true_obj = 1.7751 # not verified with SDP
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr14_JuMP() # out of memory error when converting sparse to dense in MOI conversion
    # (n, deg, num_points, signal_ratio, f) = (5, 5, 1000, 0.0, x -> exp(norm(x)))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true, use_dense = true))
    # build_shapeconregr_PSD(model, X, y, deg, ShapeData(n), use_lsq_obj = true)
    model = JuMP_shapeconregr14()
    JuMP.optimize!(model)
end

function test_JuMP_shapeconregr15_JuMP() # out of memory error during preprocessing
    # (n, deg, num_points, signal_ratio, f) = (5, 5, 1000, 0.0, x -> exp(norm(x)))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true, use_dense = false))
    # build_shapeconregr_PSD(model, X, y, deg, ShapeData(n), use_lsq_obj = true)
    model = JuMP_shapeconregr15()
    JuMP.optimize!(model)
end

function test_JuMP_shapeconregr16_JuMP()
    # (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^3))
    # (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    # model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    # build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false)
    model = JuMP_shapeconregr16()
    true_obj = 4.4065e-1
    solve_and_check_JuMP(model, true_obj)
end
