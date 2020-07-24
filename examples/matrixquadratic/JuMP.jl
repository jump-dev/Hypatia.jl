#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

for variables X in R^{n, m} and Y in S^n:
    max tr(C*X) :
    Y - X*X' in S^n_+
    Y_ij = P_ij for (i, j) in Omega
where Omega is a set of fixed indices and P is a random PSD matrix

the nonlinear constraint Y - X*X' in S^n_+ is equivalent to
the conic constraint (Y, 0.5, X) in MatrixEpiPerSquareCone(),
and also to the larger conic constraint [I X'; X Y] in S^{n + m}_+
(see Lecture 4 of "Lectures on Convex Optimization" (2018) by Y. Nesterov)
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
using SparseArrays

struct MatrixQuadraticJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_rows::Int
    num_cols::Int
    use_matrixepipersquare::Bool # use matrixepipersquare cone, else PSD formulation
end

function build(inst::MatrixQuadraticJuMP{T}) where {T <: Float64} # TODO generic reals
    (num_rows, num_cols) = (inst.num_rows, inst.num_cols)
    C = randn(num_cols, num_rows)
    P = randn(num_rows, num_rows)
    P = Symmetric(P * P')
    (row_idxs, col_idxs, _) = findnz(tril!(sprand(Bool, num_rows, num_rows, inv(sqrt(num_rows)))) + I)

    model = JuMP.Model()
    JuMP.@variable(model, X[1:num_rows, 1:num_cols])
    JuMP.@variable(model, Y[1:num_rows, 1:num_rows], Symmetric)
    JuMP.@objective(model, Max, tr(C * X))
    JuMP.@constraint(model, [(row, col) in zip(row_idxs, col_idxs)], Y[row, col] == P[row, col])

    if inst.use_matrixepipersquare
        U_svec = zeros(JuMP.GenericAffExpr{Float64, JuMP.VariableRef}, Cones.svec_length(num_rows))
        U_svec = Cones.smat_to_svec!(U_svec, 1.0 * Y, sqrt(2))
        JuMP.@constraint(model, vcat(U_svec, 0.5, vec(X)) in Hypatia.MatrixEpiPerSquareCone{Float64, Float64}(num_rows, num_cols))
    else
        JuMP.@constraint(model, Symmetric([Matrix(I, num_cols, num_cols) X'; X Y]) in JuMP.PSDCone())
    end

    return model
end

instances[MatrixQuadraticJuMP]["minimal"] = [
    ((2, 2, true),),
    ((2, 2, false),),
    ]
instances[MatrixQuadraticJuMP]["fast"] = [
    ((2, 3, true),),
    ((2, 3, false),),
    ((5, 6, true),),
    ((5, 6, false),),
    ((10, 20, true),),
    ((10, 20, false),),
    ((20, 40, true),),
    ((20, 40, false),),
    ]
instances[MatrixQuadraticJuMP]["slow"] = [
    ((60, 80, true),),
    ((60, 80, false),),
    ]
