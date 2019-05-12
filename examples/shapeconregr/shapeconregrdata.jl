#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors
=#

function getshapeconregrdata(inst::Int)
    if inst == 1
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 0.0, x -> exp(norm(x)), false)
        shapedata = ShapeData(n)
        true_obj = 4.4065e-1
    elseif inst == 2
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 0.0, x -> sum(x.^3), false)
        shapedata = ShapeData(n)
        true_obj = 1.3971e-1
    elseif inst == 3
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 0.0, x -> sum(x.^4), false)
        shapedata = ShapeData(n)
        true_obj = 2.4577e-1
    elseif inst == 4
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 50.0, x -> sum(x.^3), false)
        shapedata = ShapeData(n)
        true_obj = 1.5449e-1
    elseif inst == 5
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 50.0, x -> sum(x.^4), false)
        shapedata = ShapeData(n)
        true_obj = 2.5200e-1
    elseif inst == 6
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 0.0, x -> exp(norm(x)), true)
        shapedata = ShapeData(n)
        true_obj = 5.4584e-2
    elseif inst == 7
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 50.0, x -> sum(x.^4), true)
        shapedata = ShapeData(n)
        true_obj = 3.3249e-2
    elseif inst == 8
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 4, 100, 0.0, x -> -inv(1 + exp(-10.0 * norm(x))), true)
        shapedata = ShapeData(MU.Box(zeros(n), ones(n)), MU.Box(zeros(n), ones(n)), ones(n), 1)
        true_obj = 3.7723e-03
    elseif inst == 9
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 4, 100, 10.0, x -> -inv(1 + exp(-10.0 * norm(x))), true)
        shapedata = ShapeData(MU.Box(zeros(n), ones(n)), MU.Box(zeros(n), ones(n)), ones(n), 1)
        true_obj = 3.0995e-02 # not verified with SDP
    elseif inst == 10
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 4, 100, 0.0, x -> exp(norm(x)), true)
        shapedata = ShapeData(n)
        true_obj = 5.0209e-02 # not verified with SDP
    elseif inst == 11
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 5, 100, 10.0, x -> exp(norm(x)), true)
        shapedata = ShapeData(MU.Box(0.5 * ones(n), 2 * ones(n)), MU.Box(0.5 * ones(n), 2 * ones(n)), ones(n), 1)
        true_obj = 0.22206 # not verified with SDP
    elseif inst == 12
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 6, 100, 1.0, x -> exp(norm(x)), true)
        shapedata = ShapeData(MU.Box(0.5 * ones(n), 2 * ones(n)), MU.Box(0.5 * ones(n), 2 * ones(n)), ones(n), 1)
        true_obj = 0.22206
    elseif inst == 13
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 6, 100, 1.0, x -> exp(norm(x)), false)
        shapedata = ShapeData(n)
        true_obj = 1.7751 # not verified with SDP
    elseif inst == 14
        # either out of memory error when converting sparse to dense in MOI conversion, or during preprocessing
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (5, 5, 1000, 0.0, x -> exp(norm(x)), true)
        shapedata = ShapeData(n)
        true_obj = NaN # unknown
    elseif inst == 15
        (n, deg, num_points, signal_ratio, f, use_lsq_obj) = (2, 3, 100, 0.0, x -> exp(norm(x)), false)
        shapedata = ShapeData(n)
        true_obj = 4.4065e-1
    else
        error("instance $inst not recognized")
    end
    return (n, deg, num_points, signal_ratio, f, shapedata, use_lsq_obj, true_obj)
end
