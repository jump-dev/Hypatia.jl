#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

problem 1 (linear_dynamics = true)
eigenvalue problem related to Lyapunov stability example from sections 2.2.2 / 6.3.2
"Linear Matrix Inequalities in System and Control Theory" by
S. Boyd, L. El Ghaoui, E. Feron, and V. Balakrishnan:
min t
P in S_+
[-A'*P - P*A - C'C, P*B;
B'*P, tI] in S_+
for the system x_dot = A*x

problem 2
Lyapunov stability example from https://stanford.edu/class/ee363/sessions/s4notes.pdf:
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
const CO = Hypatia.Cones

function lyapunovstabilityJuMP(
    side::Int;
    use_matrixepipersquare::Bool = true,
    linear_dynamics::Bool = true,
    )
    model = JuMP.Model()
    JuMP.@variable(model, t)
    if linear_dynamics
        A = randn(side, side)
        A = -A * A'
        B = randn(side, side)
        B = B * B'
        C = randn(side, side) ./ 10
        JuMP.@variable(model, P[1:side, 1:side], PSD)
        U = -A' * P .- P * A .- C' * C
        W = P * B
    else
        A = randn(side, side)
        # this means P = -A is a feasible solution, with alpha and gamma sufficiently small
        A = -A * A' - I
        alpha = 0.01
        gamma = 0.01
        JuMP.@variable(model, P[1:side, 1:side], Symmetric)
        JuMP.@constraint(model, Symmetric(P - I) in JuMP.PSDCone())
        U = -A' * P .- P * A .- alpha * P .- t * gamma ^ 2
        W = -P
    end

    if use_matrixepipersquare
        U_svec = CO.smat_to_svec!(zeros(eltype(U), CO.svec_length(side)), U, sqrt(2))
        JuMP.@constraint(model, vcat(U_svec, t / 2, vec(W)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(side, side))
    else
        JuMP.@constraint(model, [U W; W' t .* Matrix(I, side, side)] in JuMP.PSDCone())
    end
    JuMP.@objective(model, Min, t)

    return (model = model,)
end

lyapunovstabilityJuMP1() = lyapunovstabilityJuMP(5, use_matrixepipersquare = true)
lyapunovstabilityJuMP2() = lyapunovstabilityJuMP(5, use_matrixepipersquare = false)
lyapunovstabilityJuMP3() = lyapunovstabilityJuMP(10, use_matrixepipersquare = true)
lyapunovstabilityJuMP4() = lyapunovstabilityJuMP(10, use_matrixepipersquare = false)
lyapunovstabilityJuMP5() = lyapunovstabilityJuMP(25, use_matrixepipersquare = true)
lyapunovstabilityJuMP6() = lyapunovstabilityJuMP(25, use_matrixepipersquare = false)
lyapunovstabilityJuMP7() = lyapunovstabilityJuMP(5, use_matrixepipersquare = true, linear_dynamics = false)
lyapunovstabilityJuMP8() = lyapunovstabilityJuMP(5, use_matrixepipersquare = false, linear_dynamics = false)
lyapunovstabilityJuMP9() = lyapunovstabilityJuMP(10, use_matrixepipersquare = true, linear_dynamics = false)
lyapunovstabilityJuMP10() = lyapunovstabilityJuMP(10, use_matrixepipersquare = false, linear_dynamics = false)
lyapunovstabilityJuMP11() = lyapunovstabilityJuMP(25, use_matrixepipersquare = true, linear_dynamics = false)
lyapunovstabilityJuMP12() = lyapunovstabilityJuMP(25, use_matrixepipersquare = false, linear_dynamics = false)

function test_lyapunovstabilityJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_lyapunovstabilityJuMP_all(; options...) = test_lyapunovstabilityJuMP.([
    lyapunovstabilityJuMP1,
    lyapunovstabilityJuMP2,
    lyapunovstabilityJuMP3,
    lyapunovstabilityJuMP4,
    lyapunovstabilityJuMP5,
    lyapunovstabilityJuMP6,
    lyapunovstabilityJuMP7,
    lyapunovstabilityJuMP8,
    lyapunovstabilityJuMP9,
    lyapunovstabilityJuMP10,
    lyapunovstabilityJuMP11,
    lyapunovstabilityJuMP12,
    ], options = options)

test_lyapunovstabilityJuMP(; options...) = test_lyapunovstabilityJuMP.([
    lyapunovstabilityJuMP1,
    lyapunovstabilityJuMP2,
    lyapunovstabilityJuMP6,
    lyapunovstabilityJuMP7,
    ], options = options)
