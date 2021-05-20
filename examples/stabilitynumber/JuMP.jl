#=
strengthening of the theta function towards the stability number of a graph

TODO add sparse PSD formulation
=#

using SparseArrays

struct StabilityNumber{T <: Real} <: ExampleInstanceJuMP{T}
    side::Int
    use_doublynonnegativetri::Bool
end

function build(inst::StabilityNumber{T}) where {T <: Float64}
    side = inst.side
    sparsity = 1 - inv(side)
    inv_graph = tril!(sprand(Bool, side, side, sparsity) + I)
    (row_idxs, col_idxs, _) = findnz(inv_graph)
    diags = (row_idxs .== col_idxs)

    model = JuMP.Model()
    JuMP.@variable(model, X[1:length(row_idxs)])
    X_diag = X[diags]
    JuMP.@objective(model, Max, 2 * sum(X) - sum(X_diag))
    JuMP.@constraint(model, sum(X_diag) == 1)
    X_lifted = sparse(row_idxs, col_idxs, X, side, side)
    X_vec = JuMP.AffExpr[X_lifted[i, j] for i in 1:side for j in 1:i]
    if inst.use_doublynonnegativetri
        cone_dim = length(X_vec)
        X_scal = Cones.scale_svec!(X_vec, sqrt(T(2)))
        JuMP.@constraint(model, X_scal in
            Hypatia.DoublyNonnegativeTriCone{T}(cone_dim))
    else
        JuMP.@constraint(model, X_vec in
            MOI.PositiveSemidefiniteConeTriangle(side))
        JuMP.@constraint(model, X[.!(diags)] .>= 0)
    end

    return model
end
