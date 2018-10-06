#=
Copyright 2018, Chris Coey and contributors
=#

using Hypatia
using Test

verbflag = true # test verbosity

# TODO interpolation tests

# native interface tests
# TODO test all interface functions (codecov will help)
# include(joinpath(@__DIR__, "native.jl"))
# testnative(verbflag, Hypatia.QRCholCache)
# # testnative(verbflag, Hypatia.QRConjGradCache)
# testnative(verbflag, Hypatia.NaiveCache)

# MathOptInterface tests
# TODO test with a variety of methods/options (eg various linsys solvers)
include(joinpath(@__DIR__, "moi.jl"))
# testmoi(verbflag, true)
# testmoi(verbflag, false) # TODO fails on empty sparse A


optimizer = MOIU.CachingOptimizer(HypatiaModelData{Float64}(), Hypatia.HypatiaOptimizer(usedense=true))

config = MOIT.TestConfig(
    atol=1e-3,
    rtol=1e-3,
    solve=true,
    query=true,
    modify_lhs=true,
    duals=true,
    infeas_certificates=true,
    )


function linear15test(model::MOI.ModelLike, config::MOIT.TestConfig)
    atol = config.atol
    rtol = config.rtol
    # minimize 0
    # s.t. 0 == 0
    #      x == 1
    @test MOI.supports(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    @test MOI.supports(model, MOI.ObjectiveSense())
    @test MOI.supports_constraint(model, MOI.VectorAffineFunction{Float64}, MOI.Zeros)

    MOI.empty!(model)
    @test MOI.is_empty(model)

    x = MOI.add_variables(model, 1)
    # Create a VectorAffineFunction with two rows, but only
    # one term, belonging to the second row. The first row,
    # which is empty, is essentially a constraint that 0 == 0.
    c = MOI.add_constraint(model,
        MOI.VectorAffineFunction(
            MOI.VectorAffineTerm.(2, MOI.ScalarAffineTerm.([1.0], x)),
            zeros(2)
        ),
        MOI.Zeros(2)
    )

    MOI.set(model,
        MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
        MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0], x), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MinSense)

    if config.solve
        MOI.optimize!(model)

        @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success

        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint

        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 0 atol=atol rtol=rtol

        @test MOI.get(model, MOI.VariablePrimal(), x[1]) ≈ 0 atol=atol rtol=rtol
    end
end

function linear11test(model::MOI.ModelLike, config::MOIT.TestConfig)
    atol = config.atol
    rtol = config.rtol
    # simple 2 variable, 1 constraint problem
    # min x + y
    # st   x + y >= 1
    #      x + y >= 2

    @test MOI.supports(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    @test MOI.supports(model, MOI.ObjectiveSense())
    @test MOI.supports_constraint(model, MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64})
    @test MOI.supports_constraint(model, MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64})

    MOI.empty!(model)
    @test MOI.is_empty(model)

    v = MOI.add_variables(model, 2)

    c1 = MOI.add_constraint(model, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], v), 0.0), MOI.GreaterThan(1.0))
    c2 = MOI.add_constraint(model, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], v), 0.0), MOI.GreaterThan(2.0))

    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], v), 0.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MinSense)

    if config.solve
        MOI.optimize!(model)

        @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 2.0 atol=atol rtol=rtol
    end

    c3 = MOI.transform(model, c2, MOI.LessThan(2.0))

    @test isa(c3, MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}})
    @test MOI.is_valid(model, c2) == false
    @test MOI.is_valid(model, c3) == true

    if config.solve
        MOI.optimize!(model)

        @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
        @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1.0 atol=atol rtol=rtol
    end
end

function linear2test(model::MOI.ModelLike, config::MOIT.TestConfig)
    atol = config.atol
    rtol = config.rtol
    # Min -x
    # s.t. x + y <= 1
    # x, y >= 0

    @test MOI.supports(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    @test MOI.supports(model, MOI.ObjectiveSense())
    @test MOI.supports_constraint(model, MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64})
    @test MOI.supports_constraint(model, MOI.SingleVariable, MOI.GreaterThan{Float64})

    MOI.empty!(model)
    @test MOI.is_empty(model)

    x = MOI.add_variable(model)
    y = MOI.add_variable(model)

    @test MOI.get(model, MOI.NumberOfVariables()) == 2

    cf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], [x, y]), 0.0)
    c = MOI.add_constraint(model, cf, MOI.LessThan(1.0))
    @test MOI.get(model, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}()) == 1

    vc1 = MOI.add_constraint(model, MOI.SingleVariable(x), MOI.GreaterThan(0.0))
    vc2 = MOI.add_constraint(model, MOI.SingleVariable(y), MOI.GreaterThan(0.0))
    @test MOI.get(model, MOI.NumberOfConstraints{MOI.SingleVariable,MOI.GreaterThan{Float64}}()) == 2

    objf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0,0.0], [x, y]), 0.0)
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objf)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MinSense)

    @test MOI.get(model, MOI.ObjectiveSense()) == MOI.MinSense

    if config.solve
        MOI.optimize!(model)

        @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success

        @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint

        @test MOI.get(model, MOI.ObjectiveValue()) ≈ -1 atol=atol rtol=rtol

        @test MOI.get(model, MOI.VariablePrimal(), x) ≈ 1 atol=atol rtol=rtol

        @test MOI.get(model, MOI.VariablePrimal(), y) ≈ 0 atol=atol rtol=rtol

        @test MOI.get(model, MOI.ConstraintPrimal(), c) ≈ 1 atol=atol rtol=rtol

        if config.duals
            @test MOI.get(model, MOI.DualStatus()) == MOI.FeasiblePoint
            @test MOI.get(model, MOI.ConstraintDual(), c) ≈ -1 atol=atol rtol=rtol

            # reduced costs
            @test MOI.get(model, MOI.ConstraintDual(), vc1) ≈ 0 atol=atol rtol=rtol
            @test MOI.get(model, MOI.ConstraintDual(), vc2) ≈ 1 atol=atol rtol=rtol
        end
    end
end



@testset begin
    # linear2test(optimizer, config)
    # linear15test(optimizer, config)
    linear11test(optimizer, config)
end



return nothing
