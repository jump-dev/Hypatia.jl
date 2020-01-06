#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

Lyapunov stability example from https://stanford.edu/class/ee363/sessions/s4notes.pdf

min t
P - I in S_+
[A'*P + P*A + alpha*P + t*gamma^2*I, P;
P, -tI] in S_-

originally a feasibility problem, a feasible P and t prove the existence of a Lyapunov function
for the system x_dot = A*x+g(x), norm(g(x)) <= gamma*norm(x)
=#

using LinearAlgebra
import Random
using Test
import MathOptInterface
const MOI = MathOptInterface
import JuMP
import Hypatia
const MU = Hypatia.ModelUtilities

function lyapunovJuMP(
    side::Int;
    use_matrixepipersquare::Bool = true,
    )
    A = randn(side, side)
    # this means P = -A is a feasible solution, with alpha and gamma sufficiently small
    A = -A * A' - I
    alpha = 0.01
    gamma = 0.01

    model = JuMP.Model()
    JuMP.@variable(model, P[1:side, 1:side], Symmetric)
    JuMP.@variable(model, t >= 0)
    U = A' * P .+ P * A .+ alpha * P .+ t * gamma ^ 2
    JuMP.@constraint(model, Symmetric(P - I) in JuMP.PSDCone())
    if use_matrixepipersquare
        U_vec = [U[i, j] for i in 1:side for j in 1:i]
        MU.vec_to_svec!(U_vec)
        JuMP.@constraint(model, vcat(-U_vec, t / 2, vec(-P)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(side, side))
    else
        JuMP.@constraint(model, [-U -P; -P t .* Matrix(I, side, side)] in JuMP.PSDCone())
    end
    JuMP.@objective(model, Min, t)

    return (model = model,)
end

lyapunovJuMP1() = lyapunovJuMP(5, use_matrixepipersquare = true)
lyapunovJuMP2() = lyapunovJuMP(5, use_matrixepipersquare = false)
lyapunovJuMP3() = lyapunovJuMP(10, use_matrixepipersquare = true)
lyapunovJuMP4() = lyapunovJuMP(10, use_matrixepipersquare = false)
lyapunovJuMP5() = lyapunovJuMP(25, use_matrixepipersquare = true)
lyapunovJuMP6() = lyapunovJuMP(25, use_matrixepipersquare = false)

function test_lyapunovJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_lyapunovJuMP_all(; options...) = test_lyapunovJuMP.([
    lyapunovJuMP1,
    lyapunovJuMP2,
    lyapunovJuMP3,
    lyapunovJuMP4,
    lyapunovJuMP5,
    lyapunovJuMP6,
    ], options = options)

test_lyapunovJuMP(; options...) = test_lyapunovJuMP.([
    lyapunovJuMP1,
    lyapunovJuMP2,
    ], options = options)
