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
(see Lecture 4, Lectures on Convex Optimization by Y. Nesterov)
and also equivalent to
(Y-C*X*D-(C*X*D)'-E, 1, A*X*B) in MatrixEpiPerSquareCone()
=#

import JuMP
import MathOptInterface
const MOI = MathOptInterface
import Hypatia
CO = Hypatia.Cones
import Random
using LinearAlgebra
using Test

function matrixquadraticJuMP(
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

matrixquadraticJuMP1() = matrixquadraticJuMP(5, 6, use_matrixepipersquare = true)
matrixquadraticJuMP2() = matrixquadraticJuMP(5, 6, use_matrixepipersquare = false)
matrixquadraticJuMP3() = matrixquadraticJuMP(10, 20, use_matrixepipersquare = true)
matrixquadraticJuMP4() = matrixquadraticJuMP(10, 20, use_matrixepipersquare = false)
matrixquadraticJuMP5() = matrixquadraticJuMP(25, 30, use_matrixepipersquare = true)
matrixquadraticJuMP6() = matrixquadraticJuMP(25, 30, use_matrixepipersquare = false)

function test_matrixquadraticJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_matrixquadraticJuMP_all(; options...) = test_matrixquadraticJuMP.([
    matrixquadraticJuMP1,
    matrixquadraticJuMP2,
    matrixquadraticJuMP3,
    matrixquadraticJuMP4,
    matrixquadraticJuMP5,
    matrixquadraticJuMP6,
    ], options = options)

test_matrixquadraticJuMP(; options...) = test_matrixquadraticJuMP.([
    matrixquadraticJuMP1,
    matrixquadraticJuMP2,
    matrixquadraticJuMP3,
    matrixquadraticJuMP4,
    ], options = options)
