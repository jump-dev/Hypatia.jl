using Hypatia
import MathOptInterface
const MOI = MathOptInterface
using JuMP
# import PolyJuMP
import DynamicPolynomials
const DP = DynamicPolynomials
# import SumOfSquares
using LinearAlgebra
import Random
using Test

rt2 = sqrt(2)
Random.seed!(1)

use_soc  = true


model = Model(with_optimizer(Hypatia.Optimizer, verbose = true))
vec_length = 5
nrandoms = vec_length - 1
randvals = randn(nrandoms)
@variable(model, x0)
@objective(model, Min, x0)
if use_soc
    @constraint(model, vcat(x0, randvals...) in SecondOrderCone())
else
    matrix_condition = JuMP.GenericAffExpr{Float64,VariableRef}[]
    push!(matrix_condition, x0)
    for i in 2:vec_length
        push!(matrix_condition, randvals[i - 1]...)
        for j in 2:(i - 1)
            push!(matrix_condition, 0)
        end
        push!(matrix_condition, x0)
    end
    sqrconstr = @constraint(model, matrix_condition in MOI.PositiveSemidefiniteConeTriangle(vec_length))
end
optimize!(model)
dual(sqrconstr)
