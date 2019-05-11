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
    true_obj = -1.36775
    model = JuMP_polymin1()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_polymin2()
    # the Schwefel polynomial in a box
    true_obj = -0
    model = JuMP_polymin2()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_polymin3()
    # the Magnetism polynomial in a ball
    true_obj = -0.25
    model = JuMP_polymin3()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_polymin4()
    # the Motzkin polynomial in an ellipsoid containing two local minima in opposite orthants
    true_obj = 0
    model = JuMP_polymin4()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_polymin5()
    true_obj = -3.1800966258
    model = JuMP_polymin5()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_polymin6()
    true_obj = 3
    model = JuMP_polymin6()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_polymin7()
    true_obj = -20.8
    model = JuMP_polymin7()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_polymin8()
    true_obj = 0.814814
    model =JuMP_polymin8()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_polymin9()
    true_obj = -73.31
    model = JuMP_polymin9()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_polymin10()
    true_obj = 0
    model = JuMP_polymin10()
    solve_and_check_JuMP(model, true_obj, atol = 1e-3)
end

function test_JuMP_shapeconregr1()
    true_obj = 4.4065e-1
    model = JuMP_shapeconregr1()
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr2()
    true_obj = 1.3971e-1
    model = JuMP_shapeconregr2(sample = true)
    solve_and_check_JuMP(model, true_obj)
    model = JuMP_shapeconregr2(sample = false)
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr3()
    model = JuMP_shapeconregr3()
    true_obj = 2.4577e-1
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr4()
    model = JuMP_shapeconregr4()
    true_obj = 1.5449e-1
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr5()
    model = JuMP_shapeconregr5()
    true_obj = 2.5200e-1
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr6()
    model = JuMP_shapeconregr6()
    true_obj = 5.4584e-2
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr7()
    model = JuMP_shapeconregr7()
    true_obj = 3.3249e-2
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr8()
    model = JuMP_shapeconregr8()
    true_obj = 3.7723e-03
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr9()
    model = JuMP_shapeconregr9()
    true_obj = 3.0995e-02 # not verified with SDP
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr10()
    model = JuMP_shapeconregr10()
    true_obj = 5.0209e-02 # not verified with SDP
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr11()
    model = JuMP_shapeconregr11()
    true_obj = 0.22206 # not verified with SDP
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr12() # SDP
    model = JuMP_shapeconregr12()
    true_obj = 0.22206
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr13()
    model = JuMP_shapeconregr13()
    true_obj = 1.7751 # not verified with SDP
    solve_and_check_JuMP(model, true_obj)
end

function test_JuMP_shapeconregr14() # out of memory error when converting sparse to dense in MOI conversion, SDP
    model = JuMP_shapeconregr14()
    JuMP.optimize!(model)
end

function test_JuMP_shapeconregr15() # out of memory error during preprocessing, SDP
    model = JuMP_shapeconregr15()
    JuMP.optimize!(model)
end
