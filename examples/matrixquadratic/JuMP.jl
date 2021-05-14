#=
for variables X in R^{n, m} and Y in S^n:
maximize    tr(C*X) :
subject to  Y - X*X' in S^n_+
            Y_ij = P_ij for (i, j) in Omega
where Omega is a set of fixed indices and P is a random PSD matrix

the nonlinear constraint Y - X*X' in S^n_+ is equivalent to
the conic constraint (Y, 0.5, X) in MatrixEpiPerSquareCone(),
and also to the larger conic constraint [I X'; X Y] in S^{n + m}_+

simple case of nonlinfear matrix inequality example from Lectures on Modern
Convex Optimization by Aharon Ben-Tal and Arkadi Nemirovski, pg 154
=#

using SparseArrays

struct MatrixQuadraticJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_rows::Int
    num_cols::Int
    use_matrixepipersquare::Bool # use matrixepipersquare cone, else PSD cone
end

function build(inst::MatrixQuadraticJuMP{T}) where {T <: Float64}
    (num_rows, num_cols) = (inst.num_rows, inst.num_cols)
    C = randn(num_cols, num_rows)
    P = randn(num_rows, num_rows)
    P = Symmetric(P * P')
    (row_idxs, col_idxs, _) = findnz(tril!(sprand(Bool, num_rows,
        num_rows, inv(sqrt(num_rows)))) + I)

    model = JuMP.Model()
    JuMP.@variable(model, X[1:num_rows, 1:num_cols])
    JuMP.@variable(model, Y[1:num_rows, 1:num_rows], Symmetric)
    JuMP.@objective(model, Max, tr(C * X))
    JuMP.@constraint(model, [(row, col) in zip(row_idxs, col_idxs)],
        Y[row, col] == P[row, col])

    if inst.use_matrixepipersquare
        U_svec = zeros(JuMP.GenericAffExpr{T, JuMP.VariableRef},
            Cones.svec_length(num_rows))
        U_svec = Cones.smat_to_svec!(U_svec, 1.0 * Y, sqrt(2))
        JuMP.@constraint(model, vcat(U_svec, 0.5, vec(X)) in
            Hypatia.MatrixEpiPerSquareCone{T, T}(num_rows, num_cols))
    else
        JuMP.@constraint(model, Symmetric(
            [Matrix(I, num_cols, num_cols) X'; X Y]) in JuMP.PSDCone())
    end

    return model
end
