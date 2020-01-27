#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

minimize the trace of the epigraph of a matrix quadratic function
max tr(CX)
s.t. Y - X*X' in S_+
tr(1*Y) = 1
Y_ij = P_ij for (i, j) in Omega

Y - A*X*B*B'*X'*C - C*X*D - (C*X*D)' - E in S_+ equivalent to
[
I  X'
X  Y
]
in S_+
(see Lecture 4, Lectures on Convex Optimization by Y. Nesterov)
and also equivalent to
(Y, 0.5, X) in MatrixEpiPerSquareCone()
=#

import JuMP
import MathOptInterface
const MOI = MathOptInterface
import Hypatia
CO = Hypatia.Cones
import Random
using LinearAlgebra
using Test

function matquadraticJuMP(
    W_rows::Int,
    W_cols::Int;
    use_matrixepipersquare::Bool = true,
    )
    C = randn(W_cols, W_rows)
    num_fixed = div(W_rows, 2)
    fixed_rows = rand(1:W_rows, num_fixed)
    fixed_cols = rand(1:W_rows, num_fixed)
    fixed_vals = randn(num_fixed)
    for i in 1:num_fixed
        if fixed_rows[i] > fixed_cols[i]
            (fixed_rows[i], fixed_cols[i]) = (fixed_cols[i], fixed_rows[i])
        end
    end

    model = JuMP.Model()
    JuMP.@variable(model, X[1:W_rows, 1:W_cols])
    JuMP.@variable(model, Y[1:W_rows, 1:W_rows], Symmetric)
    JuMP.@objective(model, Max, tr(C * X))

    if use_matrixepipersquare
        # U = Symmetric(Y)
        U_svec = CO.smat_to_svec!(zeros(eltype(Y.data .* 1), CO.svec_length(W_rows)), Y.data .* 1, sqrt(2))
        JuMP.@constraint(model, vcat(U_svec, 0.5, vec(X)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(W_rows, W_cols))
    else
        JuMP.@constraint(model, Symmetric([Matrix(I, W_cols, W_cols) X'; X Y]) in JuMP.PSDCone())
    end
    JuMP.@constraint(model, sum(Y) == 1)
    JuMP.@constraint(model, [k in 1:num_fixed], Y[fixed_rows[k], fixed_cols[k]]  == fixed_vals[k])

    return (model = model,)
end

matquadraticJuMP1() = matquadraticJuMP(5, 6, use_matrixepipersquare = true)
matquadraticJuMP2() = matquadraticJuMP(5, 6, use_matrixepipersquare = false)
matquadraticJuMP3() = matquadraticJuMP(10, 20, use_matrixepipersquare = true)
matquadraticJuMP4() = matquadraticJuMP(10, 20, use_matrixepipersquare = false)
matquadraticJuMP5() = matquadraticJuMP(25, 30, use_matrixepipersquare = true)
matquadraticJuMP6() = matquadraticJuMP(25, 30, use_matrixepipersquare = false)

function test_matquadraticJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_matquadraticJuMP_all(; options...) = test_matquadraticJuMP.([
    matquadraticJuMP1,
    matquadraticJuMP2,
    matquadraticJuMP3,
    matquadraticJuMP4,
    matquadraticJuMP5,
    matquadraticJuMP6,
    ], options = options)

test_matquadraticJuMP(; options...) = test_matquadraticJuMP.([
    matquadraticJuMP1,
    matquadraticJuMP2,
    matquadraticJuMP3,
    matquadraticJuMP4,
    ], options = options)
