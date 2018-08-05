
using MathOptInterface
const MOI = MathOptInterface
const MOIT = MOI.Test
const MOIB = MOI.Bridges
const MOIU = MOI.Utilities

MOIU.@model AlfonsoModelData () () (Zeros, Nonnegatives) () () () (VectorOfVariables,) (VectorAffineFunction,)
const optimizer = MOIU.CachingOptimizer(AlfonsoModelData{Float64}(), Alfonso.Optimizer())

const config = MOIT.TestConfig(atol=1e-4, rtol=1e-4)

# @testset "Continuous linear problems" begin
#     MOIT.contlineartest(MOIB.SplitInterval{Float64}(optimizer), config)
# end
