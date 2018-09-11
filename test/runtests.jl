
using Alfonso
using Test

# verbflag = true # Alfonso verbose option

# # TODO interpolation tests
#
# # load optimizer builder functions from examples folder
# egs_dir = joinpath(@__DIR__, "../examples")
# include(joinpath(egs_dir, "envelope/envelope.jl"))
# include(joinpath(egs_dir, "lp/lp.jl"))
# include(joinpath(egs_dir, "namedpoly/namedpoly.jl"))
#
# # run native and MOI interfaces on examples
# include(joinpath(@__DIR__, "nativeexamples.jl"))


import MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities

MOIU.@model(AlfonsoModelData,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan),
    (MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.PositiveSemidefiniteConeTriangle, MOI.ExponentialCone, MOI.PowerCone),
    (),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    )

const optimizer = MOIU.CachingOptimizer(AlfonsoModelData{Float64}(), Alfonso.Optimizer())

const config = MOIT.TestConfig(
    atol=1e-4,
    rtol=1e-4,
    solve=true,
    query=true,
    modify_lhs=true,
    duals=false,
    infeas_certificates=false,
    )

# @testset "Continuous linear problems" begin
#     MOIT.contlineartest(MOIB.SplitInterval{Float64}(optimizer), config)
# end

@testset "Continuous conic problems" begin
    exclude = ["rootdet", "logdet"]
    MOIT.contconictest(
        MOIB.SquarePSD{Float64}(
        MOIB.GeoMean{Float64}(
        # MOIB.LogDet{Float64}(
        # MOIB.RootDet{Float64}(
            optimizer
        )),
        config, exclude)
end


#
#
#
# model = optimizer
#
# atol = config.atol
# rtol = config.rtol
# # simple 2 variable, 1 constraint problem
# # min -x
# # st   x + y <= 1   (x + y - 1 ∈ Nonpositives)
# #       x, y >= 0   (x, y ∈ Nonnegatives)
#
# @test MOI.supports(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
# @test MOI.supports(model, MOI.ObjectiveSense())
# @test MOI.supports_constraint(model, MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64})
# @test MOI.supports_constraint(model, MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64})
# @test MOI.supports_constraint(model, MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64})
# @test MOI.supports_constraint(model, MOI.SingleVariable, MOI.EqualTo{Float64})
# @test MOI.supports_constraint(model, MOI.SingleVariable, MOI.GreaterThan{Float64})
#
# #@test MOI.get(model, MOI.SupportsAddConstraintAfterSolve())
# #@test MOI.get(model, MOI.SupportsAddVariableAfterSolve())
# #@test MOI.get(model, MOI.SupportsDeleteConstraint())
#
# MOI.empty!(model)
# @test MOI.is_empty(model)
#
# v = MOI.add_variables(model, 2)
# @test MOI.get(model, MOI.NumberOfVariables()) == 2
#
# cf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0], v), 0.0)
# c = MOI.add_constraint(model, cf, MOI.LessThan(1.0))
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}()) == 1
#
# vc1 = MOI.add_constraint(model, MOI.SingleVariable(v[1]), MOI.GreaterThan(0.0))
# # test fallback
# vc2 = MOI.add_constraint(model, v[2], MOI.GreaterThan(0.0))
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.SingleVariable,MOI.GreaterThan{Float64}}()) == 2
#
# # note: adding some redundant zero coefficients to catch solvers that don't handle duplicate coefficients correctly:
# objf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0,0.0,-1.0,0.0,0.0,0.0], [v; v; v]), 0.0)
# MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objf)
# MOI.set(model, MOI.ObjectiveSense(), MOI.MinSense)
#
# @test MOI.get(model, MOI.ObjectiveSense()) == MOI.MinSense
#
# if config.query
#     vrs = MOI.get(model, MOI.ListOfVariableIndices())
#     @test vrs == v || vrs == reverse(v)
#
#     @test objf ≈ MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
#
#     @test cf ≈ MOI.get(model, MOI.ConstraintFunction(), c)
#
#     s = MOI.get(model, MOI.ConstraintSet(), c)
#     @test s == MOI.LessThan(1.0)
#
#     s = MOI.get(model, MOI.ConstraintSet(), vc1)
#     @test s == MOI.GreaterThan(0.0)
#
#     s = MOI.get(model, MOI.ConstraintSet(), vc2)
#     @test s == MOI.GreaterThan(0.0)
# end
#
# if config.solve
#     MOI.optimize!(model)
#
#     @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
#
#     @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
#
#     @test MOI.get(model, MOI.ObjectiveValue()) ≈ -1 atol=atol rtol=rtol
#
#     @test MOI.get(model, MOI.VariablePrimal(), v) ≈ [1, 0] atol=atol rtol=rtol
#
#     # @test MOI.get(model, MOI.ConstraintPrimal(), c) ≈ 1 atol=atol rtol=rtol
#
#     if config.duals
#         @test MOI.get(model, MOI.DualStatus()) == MOI.FeasiblePoint
#         @test MOI.get(model, MOI.ConstraintDual(), c) ≈ -1 atol=atol rtol=rtol
#
#         # reduced costs
#         @test MOI.get(model, MOI.ConstraintDual(), vc1) ≈ 0 atol=atol rtol=rtol
#         @test MOI.get(model, MOI.ConstraintDual(), vc2) ≈ 1 atol=atol rtol=rtol
#     end
# end
#
# # change objective to Max +x
#
# objf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,0.0], v), 0.0)
# MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objf)
# MOI.set(model, MOI.ObjectiveSense(), MOI.MaxSense)
#
# if config.query
#     @test objf ≈ MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
# end
#
# @test MOI.get(model, MOI.ObjectiveSense()) == MOI.MaxSense
#
# if config.solve
#     MOI.optimize!(model)
#
#     @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
#
#     @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
#
#     @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1 atol=atol rtol=rtol
#
#     @test MOI.get(model, MOI.VariablePrimal(), v) ≈ [1, 0] atol=atol rtol=rtol
#
#     if config.duals
#         @test MOI.get(model, MOI.DualStatus()) == MOI.FeasiblePoint
#         @test MOI.get(model, MOI.ConstraintDual(), c) ≈ -1 atol=atol rtol=rtol
#
#         @test MOI.get(model, MOI.ConstraintDual(), vc1) ≈ 0 atol=atol rtol=rtol
#         @test MOI.get(model, MOI.ConstraintDual(), vc2) ≈ 1 atol=atol rtol=rtol
#     end
# end
#
# # add new variable to get :
# # max x + 2z
# # s.t. x + y + z <= 1
# # x,y,z >= 0
#
# z = MOI.add_variable(model)
# push!(v, z)
# @test v[3] == z
#
# if config.query
#     # Test that the modification of v has not affected the model
#     vars = map(t -> t.variable_index, MOI.get(model, MOI.ConstraintFunction(), c).terms)
#     @test vars == [v[1], v[2]] || vars == [v[2], v[1]]
#     @test MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, v[1])], 0.0) ≈ MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
# end
#
# vc3 = MOI.add_constraint(model, MOI.SingleVariable(v[3]), MOI.GreaterThan(0.0))
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.SingleVariable,MOI.GreaterThan{Float64}}()) == 3
#
# if config.modify_lhs
#     MOI.modify(model, c, MOI.ScalarCoefficientChange{Float64}(z, 1.0))
# else
#     MOI.delete(model, c)
#     cf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,1.0,1.0], v), 0.0)
#     c = MOI.add_constraint(model, cf, MOI.LessThan(1.0))
# end
#
# MOI.modify(model,
#     MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
#     MOI.ScalarCoefficientChange{Float64}(z, 2.0)
# )
#
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}()) == 1
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.SingleVariable,MOI.GreaterThan{Float64}}()) == 3
#
# if config.solve
#     MOI.optimize!(model)
#
#     @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
#
#     @test MOI.get(model, MOI.ResultCount()) >= 1
#
#     @test MOI.get(model, MOI.PrimalStatus(1)) == MOI.FeasiblePoint
#
#     @test MOI.get(model, MOI.ObjectiveValue()) ≈ 2 atol=atol rtol=rtol
#
#     @test MOI.get(model, MOI.VariablePrimal(), v) ≈ [0, 0, 1] atol=atol rtol=rtol
#
#     # @test MOI.get(model, MOI.ConstraintPrimal(), c) ≈ 1 atol=atol rtol=rtol
#
#     if config.duals
#         @test MOI.get(model, MOI.DualStatus()) == MOI.FeasiblePoint
#         @test MOI.get(model, MOI.ConstraintDual(), c) ≈ -2 atol=atol rtol=rtol
#
#         @test MOI.get(model, MOI.ConstraintDual(), vc1) ≈ 1 atol=atol rtol=rtol
#         @test MOI.get(model, MOI.ConstraintDual(), vc2) ≈ 2 atol=atol rtol=rtol
#         @test MOI.get(model, MOI.ConstraintDual(), vc3) ≈ 0 atol=atol rtol=rtol
#     end
# end
#
# # setting lb of x to -1 to get :
# # max x + 2z
# # s.t. x + y + z <= 1
# # x >= -1
# # y,z >= 0
# MOI.set(model, MOI.ConstraintSet(), vc1, MOI.GreaterThan(-1.0))
#
# if config.solve
#     MOI.optimize!(model)
#
#     @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
#
#     @test MOI.get(model, MOI.ResultCount()) >= 1
#
#     @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
#
#     @test MOI.get(model, MOI.ObjectiveValue()) ≈ 3 atol=atol rtol=rtol
#
#     @test MOI.get(model, MOI.VariablePrimal(), v) ≈ [-1, 0, 2] atol=atol rtol=rtol
# end
#
# # put lb of x back to 0 and fix z to zero to get :
# # max x + 2z
# # s.t. x + y + z <= 1
# # x, y >= 0, z = 0 (vc3)
# MOI.set(model, MOI.ConstraintSet(), vc1, MOI.GreaterThan(0.0))
#
# MOI.delete(model, vc3)
#
# vc3 = MOI.add_constraint(model, MOI.SingleVariable(v[3]), MOI.EqualTo(0.0))
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.SingleVariable,MOI.GreaterThan{Float64}}()) == 2
#
# if config.solve
#     MOI.optimize!(model)
#
#     @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
#
#     @test MOI.get(model, MOI.ResultCount()) >= 1
#
#     @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
#
#     @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1 atol=atol rtol=rtol
#
#     @test MOI.get(model, MOI.VariablePrimal(), v) ≈ [1, 0, 0] atol=atol rtol=rtol
# end
#
# # modify affine linear constraint set to be == 2 to get :
# # max x + 2z
# # s.t. x + y + z == 2 (c)
# # x,y >= 0, z = 0
# MOI.delete(model, c)
# # note: adding some redundant zero coefficients to catch solvers that don't handle duplicate coefficients correctly:
# cf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0,0.0,0.0,1.0,1.0,1.0,0.0,0.0,0.0], [v; v; v]), 0.0)
# c = MOI.add_constraint(model, cf, MOI.EqualTo(2.0))
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}()) == 0
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}()) == 1
#
# if config.solve
#     MOI.optimize!(model)
#
#     @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
#
#     @test MOI.get(model, MOI.ResultCount()) >= 1
#
#     @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
#
#     @test MOI.get(model, MOI.ObjectiveValue()) ≈ 2 atol=atol rtol=rtol
#
#     @test MOI.get(model, MOI.VariablePrimal(), v) ≈ [2, 0, 0] atol=atol rtol=rtol
# end
#
# # modify objective function to x + 2y to get :
# # max x + 2y
# # s.t. x + y + z == 2 (c)
# # x,y >= 0, z = 0
#
# objf = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0,2.0,0.0], v), 0.0)
# MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), objf)
# MOI.set(model, MOI.ObjectiveSense(), MOI.MaxSense)
#
# if config.query
#     @test objf ≈ MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
# end
#
# if config.solve
#     MOI.optimize!(model)
#
#     @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
#
#     @test MOI.get(model, MOI.ResultCount()) >= 1
#
#     @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
#
#     @test MOI.get(model, MOI.ObjectiveValue()) ≈ 4 atol=atol rtol=rtol
#
#     @test MOI.get(model, MOI.VariablePrimal(), v) ≈ [0, 2, 0] atol=atol rtol=rtol
# end
#
# # add constraint x - y >= 0 (c2) to get :
# # max x+2y
# # s.t. x + y + z == 2 (c)
# # x - y >= 0 (c2)
# # x,y >= 0 (vc1,vc2), z = 0 (vc3)
#
# cf2 = MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, -1.0, 0.0], v), 0.0)
# c2 = MOI.add_constraint(model, cf2, MOI.GreaterThan(0.0))
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.EqualTo{Float64}}()) == 1
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.GreaterThan{Float64}}()) == 1
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64},MOI.LessThan{Float64}}()) == 0
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.SingleVariable,MOI.EqualTo{Float64}}()) == 1
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.SingleVariable,MOI.GreaterThan{Float64}}()) == 2
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.SingleVariable,MOI.LessThan{Float64}}()) == 0
#
# if config.solve
#     MOI.optimize!(model)
#
#     @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
#
#     @test MOI.get(model, MOI.ResultCount()) >= 1
#
#     @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
#
#     @test MOI.get(model, MOI.ObjectiveValue()) ≈ 3 atol=atol rtol=rtol
#
#     @test MOI.get(model, MOI.VariablePrimal(), v) ≈ [1, 1, 0] atol=atol rtol=rtol
#
#     # @test MOI.get(model, MOI.ConstraintPrimal(), c) ≈ 2 atol=atol rtol=rtol
#     #
#     # @test MOI.get(model, MOI.ConstraintPrimal(), c2) ≈ 0 atol=atol rtol=rtol
#     #
#     # @test MOI.get(model, MOI.ConstraintPrimal(), vc1) ≈ 1 atol=atol rtol=rtol
#     #
#     # @test MOI.get(model, MOI.ConstraintPrimal(), vc2) ≈ 1 atol=atol rtol=rtol
#     #
#     # @test MOI.get(model, MOI.ConstraintPrimal(), vc3) ≈ 0 atol=atol rtol=rtol
#
#     if config.duals
#         @test MOI.get(model, MOI.DualStatus(1)) == MOI.FeasiblePoint
#
#         @test MOI.get(model, MOI.ConstraintDual(), c) ≈ -1.5 atol=atol rtol=rtol
#         @test MOI.get(model, MOI.ConstraintDual(), c2) ≈ 0.5 atol=atol rtol=rtol
#
#         @test MOI.get(model, MOI.ConstraintDual(), vc1) ≈ 0 atol=atol rtol=rtol
#         @test MOI.get(model, MOI.ConstraintDual(), vc2) ≈ 0 atol=atol rtol=rtol
#         @test MOI.get(model, MOI.ConstraintDual(), vc3) ≈ 1.5 atol=atol rtol=rtol
#     end
# end
#
# if config.query
#     @test MOI.get(model, MOI.ConstraintFunction(), c2) ≈ cf2
# end
#
# # delete variable x to get :
# # max 2y
# # s.t. y + z == 2
# # - y >= 0
# # y >= 0, z = 0
#
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.SingleVariable,MOI.GreaterThan{Float64}}()) == 2
# MOI.delete(model, v[1])
# @test MOI.get(model, MOI.NumberOfConstraints{MOI.SingleVariable,MOI.GreaterThan{Float64}}()) == 1
#
# if config.query
#     @test MOI.get(model, MOI.ConstraintFunction(), c2) ≈ MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([-1.0, 0.0], [v[2], z]), 0.0)
#
#     vrs = MOI.get(model, MOI.ListOfVariableIndices())
#     @test vrs == [v[2], z] || vrs == [z, v[2]]
#     @test MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}()) ≈ MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([2.0, 0.0], [v[2], z]), 0.0)
# end
#
#



return nothing
