#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

function JuMP_shapeconregr1(; sample = true, use_dense = true)
    (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> exp(norm(x)))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr2(; sample = true, use_dense = true)
    (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^3))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr3(; sample = true, use_dense = true)
    (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^4))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr4(; sample = true, use_dense = true)
    (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^3))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr5(; sample = true, use_dense = true)
    (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^4))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr6(; sample = true, use_dense = true)
    (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> exp(norm(x)))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = true, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr7(; sample = true, use_dense = true)
    (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 50.0, x -> sum(x.^4))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = true, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr8()
    (n, deg, num_points, signal_ratio, f) = (2, 4, 100, 0.0, x -> -inv(1 + exp(-10.0 * norm(x))))
    (X, y) = generate_regr_data(f, 0.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    shape_data = ShapeData(MU.Box(zeros(n), ones(n)), MU.Box(zeros(n), ones(n)), ones(n), 1)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, shape_data, use_lsq_obj = true, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr9()
    (n, deg, num_points, signal_ratio, f) = (2, 4, 100, 10.0, x -> -inv(1 + exp(-10.0 * norm(x))))
    (X, y) = generate_regr_data(f, 0.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    shape_data = ShapeData(MU.Box(zeros(n), ones(n)), MU.Box(zeros(n), ones(n)), ones(n), 1)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, shape_data, use_lsq_obj = true, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr10()
    (n, deg, num_points, signal_ratio, f) = (2, 4, 100, 0.0, x -> exp(norm(x)))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = true, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr11()
    (n, deg, num_points, signal_ratio, f) = (2, 5, 100, 10.0, x -> exp(norm(x)))
    (X, y) = generate_regr_data(f, 0.5, 2.0, n, num_points, signal_ratio = signal_ratio)
    shape_data = ShapeData(MU.Box(0.5 * ones(n), 2 * ones(n)), MU.Box(0.5 * ones(n), 2 * ones(n)), ones(n), 1)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, shape_data, use_lsq_obj = true, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr12()
    (n, deg, num_points, signal_ratio, f) = (2, 5, 100, 10.0, x -> exp(norm(x)))
    (X, y) = generate_regr_data(f, 0.5, 2.0, n, num_points, signal_ratio = signal_ratio)
    shape_data = ShapeData(MU.Box(0.5 * ones(n), 2 * ones(n)), MU.Box(0.5 * ones(n), 2 * ones(n)), ones(n), 1)
    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_PSD(model, X, y, deg, shape_data, use_lsq_obj = true, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr13()
    (n, deg, num_points, signal_ratio, f) = (2, 6, 100, 1.0, x -> exp(norm(x)))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr14() # out of memory error when converting sparse to dense in MOI conversion
    (n, deg, num_points, signal_ratio, f) = (5, 5, 1000, 0.0, x -> exp(norm(x)))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true, use_dense = true))
    return build_shapeconregr_PSD(model, X, y, deg, ShapeData(n), use_lsq_obj = true, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr15() # out of memory error during preprocessing
    (n, deg, num_points, signal_ratio, f) = (5, 5, 1000, 0.0, x -> exp(norm(x)))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = SumOfSquares.SOSModel(JuMP.with_optimizer(HYP.Optimizer, verbose = true, use_dense = false))
    return build_shapeconregr_PSD(model, X, y, deg, ShapeData(n), use_lsq_obj = true, sample = sample, use_dense = use_dense)
end

function JuMP_shapeconregr16()
    (n, deg, num_points, signal_ratio, f) = (2, 3, 100, 0.0, x -> sum(x.^3))
    (X, y) = generate_regr_data(f, -1.0, 1.0, n, num_points, signal_ratio = signal_ratio)
    model = JuMP.Model(JuMP.with_optimizer(HYP.Optimizer, verbose = true))
    return build_shapeconregr_WSOS(model, X, y, deg, ShapeData(n), use_lsq_obj = false, sample = sample, use_dense = use_dense)
end
