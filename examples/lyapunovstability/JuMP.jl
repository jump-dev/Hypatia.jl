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
for the system with linear dynamics x_dot = A*x

problem 2 (linear_dynamics = false)
Lyapunov stability example from https://stanford.edu/class/ee363/sessions/s4notes.pdf:
min t
P - I in S_+
[-A'*P - P*A - alpha*P - t*gamma^2*I, -P;
-P, tI] in S_+
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
    W_rows::Int,
    W_cols::Int = W_rows;
    use_matrixepipersquare::Bool = true,
    linear_dynamics::Bool = true,
    )
    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)

    if linear_dynamics
        A = randn(W_rows, W_rows)
        A = -A * A'
        B = randn(W_rows, W_cols)
        C = randn(W_rows, W_rows)
        JuMP.@variable(model, P[1:W_rows, 1:W_rows], PSD)
        U = -A' * P - P * A - C' * C / 100
        W = P * B
    else
        # P = -A is a feasible solution, with alpha and gamma sufficiently small
        A = randn(W_rows, W_rows)
        A = -A * A' - I
        alpha = 0.01
        gamma = 0.01
        JuMP.@variable(model, P[1:W_rows, 1:W_rows], Symmetric)
        JuMP.@constraint(model, Symmetric(P - I) in JuMP.PSDCone())
        U = -A' * P - P * A - alpha * P - (t * gamma ^ 2) .* Matrix(I, W_rows, W_rows)
        W = -P
    end

    if use_matrixepipersquare
        U_svec = CO.smat_to_svec!(zeros(eltype(U), CO.svec_length(W_rows)), U, sqrt(2))
        JuMP.@constraint(model, vcat(U_svec, t / 2, vec(W)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(W_rows, W_cols))
    else
        JuMP.@constraint(model, Symmetric([t .* Matrix(I, W_cols, W_cols) W'; W U]) in JuMP.PSDCone())
    end

    return (model = model,)
end

lyapunovstabilityJuMP1() = lyapunovstabilityJuMP(5, 6, use_matrixepipersquare = true)
lyapunovstabilityJuMP2() = lyapunovstabilityJuMP(5, 6, use_matrixepipersquare = false)
lyapunovstabilityJuMP3() = lyapunovstabilityJuMP(10, 20, use_matrixepipersquare = true)
lyapunovstabilityJuMP4() = lyapunovstabilityJuMP(10, 20, use_matrixepipersquare = false)
lyapunovstabilityJuMP5() = lyapunovstabilityJuMP(25, 30, use_matrixepipersquare = true)
lyapunovstabilityJuMP6() = lyapunovstabilityJuMP(25, 30, use_matrixepipersquare = false)
lyapunovstabilityJuMP7() = lyapunovstabilityJuMP(5, use_matrixepipersquare = true, linear_dynamics = false)
lyapunovstabilityJuMP8() = lyapunovstabilityJuMP(5, use_matrixepipersquare = false, linear_dynamics = false)
lyapunovstabilityJuMP9() = lyapunovstabilityJuMP(10, use_matrixepipersquare = true, linear_dynamics = false)
lyapunovstabilityJuMP10() = lyapunovstabilityJuMP(10, use_matrixepipersquare = false, linear_dynamics = false)
lyapunovstabilityJuMP11() = lyapunovstabilityJuMP(25, use_matrixepipersquare = true, linear_dynamics = false)
lyapunovstabilityJuMP12() = lyapunovstabilityJuMP(25, use_matrixepipersquare = false, linear_dynamics = false)

using TimerOutputs

function test_lyapunovstabilityJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    print_timer(JuMP.backend(d.model).optimizer.model.optimizer.solver.timer)
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
    lyapunovstabilityJuMP7,
    lyapunovstabilityJuMP8,
    ], options = options)

options = (use_infty_nbhd = true,)
test_lyapunovstabilityJuMP(; options...)
