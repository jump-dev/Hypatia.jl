#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

maximization with a matrix quadratic
max tr(CX)
s.t. Y - X*X' in S_+
P .<= Y .<= Q
with P .<= 0, Q .>= 0

Y - XX' in S_+ equivalent to
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
using SparseArrays
using Test

function matrixquadraticJuMP(
    W_rows::Int,
    W_cols::Int;
    use_matrixepipersquare::Bool = true,
    )
    C = randn(W_cols, W_rows)
    P = rand(W_rows, W_rows)
    Q = rand(W_rows, W_rows)

    model = JuMP.Model()
    JuMP.@variable(model, X[1:W_rows, 1:W_cols])
    JuMP.@variable(model, Y[1:W_rows, 1:W_rows], Symmetric)
    JuMP.@objective(model, Max, tr(C * X))

    if use_matrixepipersquare
        U_svec = CO.smat_to_svec!(zeros(eltype(Y.data .* 1), CO.svec_length(W_rows)), Y.data .* 1, sqrt(2))
        JuMP.@constraint(model, vcat(U_svec, 0.5, vec(X)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(W_rows, W_cols))
    else
        JuMP.@constraint(model, Symmetric([Matrix(I, W_cols, W_cols) X'; X Y]) in JuMP.PSDCone())
    end
    JuMP.@constraint(model, Y .+ P .>= 0)
    JuMP.@constraint(model, Q .- Y .>= 0)

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
