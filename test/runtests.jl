#=
Copyright 2018, Chris Coey and contributors
=#

using Hypatia
using Test

verbose = false # test verbosity

# native interface tests
# include(joinpath(@__DIR__, "native.jl"))
# testnative(verbose, Hypatia.QRCholCache)
# # testnative(verbose, Hypatia.QRConjGradCache) # TODO fails for some problems
# testnative(verbose, Hypatia.NaiveCache)

# MathOptInterface tests
include(joinpath(@__DIR__, "moi.jl"))
testmoi(verbose, true)
# testmoi(verbose, false) # TODO fails on empty sparse A



#
# optimizer = MOIU.CachingOptimizer(HypatiaModelData{Float64}(),
#     Hypatia.HypatiaOptimizer(verbose=true, usedense=true))
#
# config = MOIT.TestConfig(
#     atol=1e-3,
#     rtol=1e-3,
#     solve=true,
#     query=true,
#     modify_lhs=true,
#     duals=true,
#     infeas_certificates=true,
#     )
#
#
#
# function linear15test(model::MOI.ModelLike, config::MOIT.TestConfig)
#     atol = config.atol
#     rtol = config.rtol
#     # minimize 0
#     # s.t. 0 == 0
#     #      x == 1
#     @test MOI.supports(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
#     @test MOI.supports(model, MOI.ObjectiveSense())
#     @test MOI.supports_constraint(model, MOI.VectorAffineFunction{Float64}, MOI.Zeros)
#
#     MOI.empty!(model)
#     @test MOI.is_empty(model)
#
#     x = MOI.add_variables(model, 1)
#     # Create a VectorAffineFunction with two rows, but only
#     # one term, belonging to the second row. The first row,
#     # which is empty, is essentially a constraint that 0 == 0.
#     c = MOI.add_constraint(model,
#         MOI.VectorAffineFunction(
#             MOI.VectorAffineTerm.(2, MOI.ScalarAffineTerm.([1.0], x)),
#             zeros(2)
#         ),
#         MOI.Zeros(2)
#     )
#
#     MOI.set(model,
#         MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
#         MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([0.0], x), 0.0))
#     MOI.set(model, MOI.ObjectiveSense(), MOI.MinSense)
#
#     if config.solve
#         MOI.optimize!(model)
#
#         @test MOI.get(model, MOI.TerminationStatus()) == MOI.Success
#
#         @test MOI.get(model, MOI.PrimalStatus()) == MOI.FeasiblePoint
#
#         @test MOI.get(model, MOI.ObjectiveValue()) ≈ 0 atol=atol rtol=rtol
#
#         @test MOI.get(model, MOI.VariablePrimal(), x[1]) ≈ 0 atol=atol rtol=rtol
#     end
# end
#
# linear15test(optimizer, config)


return nothing
