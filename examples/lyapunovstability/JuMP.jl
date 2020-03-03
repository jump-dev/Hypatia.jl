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
import JuMP
const MOI = JuMP.MOI
import Hypatia
const CO = Hypatia.Cones

function lyapunovstability_JuMP(
    T::Type{Float64}, # TODO support generic reals
    W_rows::Int,
    W_cols::Int,
    linear_dynamics::Bool, # solve problem 1 in the description, else problem 2
    use_matrixepipersquare::Bool, # use matrixepipersquare cone, else PSD formulation
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
        @assert W_rows == W_cols
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

function test_lyapunovstability_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = lyapunovstability_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

lyapunovstability_JuMP_fast = [
    (5, 6, true, true),
    (5, 6, true, false),
    (5, 5, false, true),
    (5, 5, false, false),
    (10, 20, true, true),
    (10, 20, true, false),
    (15, 15, false, true),
    (15, 15, false, false),
    (25, 30, true, false),
    (30, 30, false, false),
    ]
lyapunovstability_JuMP_slow = [
    (25, 30, true, true),
    (30, 30, false, true),
    ]
