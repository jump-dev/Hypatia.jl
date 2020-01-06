#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

eigenvalue problem relating to Lyapunov stability example from sections 2.2.2 / 6.3.2
"Linear Matrix Inequalities in System and Control Theory"  by
S. Boyd, L. El Ghaoui, E. Feron, and V. Balakrishnan

min t
P in S_+
[-A'*P - P*A - C'C, P*B;
B'*P, tI] in S_+
=#

using LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia
const MU = Hypatia.ModelUtilities

function eigenprobJuMP(
    side::Int;
    use_matrixepipersquare::Bool = true,
    )
    A = randn(side, side)
    B = randn(side, side)
    C = randn(side, side)

    model = JuMP.Model()
    JuMP.@variable(model, P[1:side, 1:side], PSD)
    JuMP.@variable(model, t)
    U = -A' * P .- P * A .- C' * C
    W = P * B
    if use_matrixepipersquare
        U_vec = [U[i, j] for i in 1:side for j in 1:i]
        MU.vec_to_svec!(U_vec)
        JuMP.@constraint(model, vcat(U_vec, t / 2, vec(W)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(side, side))
    else
        JuMP.@constraint(model, [U W; W' t .* Matrix(I, side, side)] in JuMP.PSDCone())
    end
    JuMP.@objective(model, Min, t)

    return (model = model,)
end

eigenprobJuMP1() = eigenprobJuMP(5, use_matrixepipersquare = true)
eigenprobJuMP2() = eigenprobJuMP(5, use_matrixepipersquare = false)
eigenprobJuMP3() = eigenprobJuMP(10, use_matrixepipersquare = true)
eigenprobJuMP4() = eigenprobJuMP(10, use_matrixepipersquare = false)
eigenprobJuMP5() = eigenprobJuMP(25, use_matrixepipersquare = true)
eigenprobJuMP6() = eigenprobJuMP(25, use_matrixepipersquare = false)

function test_eigenprobJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_eigenprobJuMP_all(; options...) = test_eigenprobJuMP.([
    eigenprobJuMP1,
    eigenprobJuMP2,
    eigenprobJuMP3,
    eigenprobJuMP4,
    eigenprobJuMP5,
    eigenprobJuMP6,
    ], options = options)

test_eigenprobJuMP(; options...) = test_eigenprobJuMP.([
    eigenprobJuMP1,
    eigenprobJuMP2,
    ], options = options)
