#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

minimize the trace of the epigraph of a matrix quadratic function
min tr(Y)
s.t. Y - A*X*B*B'*X'*A' - C*X*D - (C*X*D)' - E in S_+
X .>= Q

Y - A*X*B*B'*X'*C - C*X*D - (C*X*D)' - E in S_+ equivalent to
[
I      (A*X*B)'
A*X*B  Y-C*X*D-(C*X*D)'-E
]
in S_+
reformulation from Lecture 4 Lectures on Convex Programming
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
    A = randn(W_rows, W_rows)
    B = randn(W_rows, W_cols)
    C = randn(W_rows, W_rows)
    D = randn(W_rows, W_rows)
    E = randn(W_rows, W_rows)
    E = E + E'

    model = JuMP.Model()
    JuMP.@variable(model, X[1:W_rows, 1:W_rows])
    JuMP.@variable(model, Y[1:W_rows, 1:W_rows], Symmetric)
    JuMP.@objective(model, Min, tr(Y))

    U = Symmetric(Y - C * X * D - (C * X * D)' - E)
    W = A * X * B
    if use_matrixepipersquare
        U_svec = CO.smat_to_svec!(zeros(eltype(U), CO.svec_length(W_rows)), U, sqrt(2))
        JuMP.@constraint(model, vcat(U_svec, 0.5, vec(W)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(W_rows, W_cols))
    else
        JuMP.@constraint(model, Symmetric([Matrix(I, W_cols, W_cols) W'; W U]) in JuMP.PSDCone())
    end
    JuMP.@constraint(model, X .>= randn(W_rows, W_rows))

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
