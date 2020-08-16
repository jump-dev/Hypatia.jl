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

struct LyapunovStabilityJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_rows::Int
    num_cols::Int
    linear_dynamics::Bool # solve problem 1 in the description, else problem 2
    use_matrixepipersquare::Bool # use matrixepipersquare cone, else PSD formulation
end

function build(inst::LyapunovStabilityJuMP{T}) where {T <: Float64} # TODO generic reals
    (num_rows, num_cols) = (inst.num_rows, inst.num_cols)

    model = JuMP.Model()
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)

    if inst.linear_dynamics
        A = randn(num_rows, num_rows)
        A = -A * A'
        B = randn(num_rows, num_cols)
        C = randn(num_rows, num_rows)
        JuMP.@variable(model, P[1:num_rows, 1:num_rows], PSD)
        U = -A' * P - P * A - C' * C / 100
        W = P * B
    else
        @assert num_rows == num_cols
        # P = -A is a feasible solution, with alpha and gamma sufficiently small
        A = randn(num_rows, num_rows)
        A = -A * A' - I
        alpha = 0.01
        gamma = 0.01
        JuMP.@variable(model, P[1:num_rows, 1:num_rows], Symmetric)
        JuMP.@constraint(model, Symmetric(P - I) in JuMP.PSDCone())
        U = -A' * P - P * A - alpha * P - (t * gamma ^ 2) .* Matrix(I, num_rows, num_rows)
        W = -P
    end

    if inst.use_matrixepipersquare
        U_svec = Cones.smat_to_svec!(zeros(eltype(U), Cones.svec_length(num_rows)), U, sqrt(2))
        JuMP.@constraint(model, vcat(U_svec, t / 2, vec(W)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(num_rows, num_cols))
    else
        JuMP.@constraint(model, Symmetric([t .* Matrix(I, num_cols, num_cols) W'; W U]) in JuMP.PSDCone())
    end

    return model
end

insts[LyapunovStabilityJuMP]["minimal"] = [
    ((2, 3, true, true),),
    ((2, 3, true, false),),
    ((2, 2, false, true),),
    ((2, 2, false, false),),
    ]
insts[LyapunovStabilityJuMP]["fast"] = [
    ((5, 6, true, true),),
    ((5, 6, true, false),),
    ((5, 5, false, true),),
    ((5, 5, false, false),),
    ((10, 20, true, true),),
    ((10, 20, true, false),),
    ((15, 15, false, true),),
    ((15, 15, false, false),),
    ((30, 30, false, false),),
    ((30, 30, false, true),),
    ]
insts[LyapunovStabilityJuMP]["slow"] = [
    ((50, 50, false, false),),
    ((50, 50, false, true),),
    ]
