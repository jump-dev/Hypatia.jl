#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

strengthening of the theta function towards the stability number of a graph

TODO
add sparse PSD formulation
=#

using SparseArrays
import Hypatia.ModelUtilities.vec_to_svec!

struct StabilityNumber{T <: Real} <: ExampleInstanceJuMP{T}
    side::Int
    use_dnn::Bool
end

function build(inst::StabilityNumber{T}) where {T <: Float64} # TODO generic reals
    (side, use_dnn) = (inst.side, inst.use_dnn)
    sparsity = 1 - 1.0 / side
    inv_graph = tril!(sprandn(side, side, sparsity)) + Diagonal(ones(side))
    (row_idxs, col_idxs, A_vals) = findnz(inv_graph)
    diag_idxs = findall(row_idxs .== col_idxs)

    model = JuMP.Model()
    JuMP.@variable(model, X[1:length(row_idxs)])
    JuMP.@objective(model, Max, 2 * sum(X) - sum(X[diag_idxs]))
    JuMP.@constraint(model, sum(X[diag_idxs]) == 1)
    X_lifted = sparse(row_idxs, col_idxs, X, side, side)
    X_vec = [X_lifted[i, j] for i in 1:side for j in 1:i]
    if inst.use_dnn
        cone_dim = length(X_vec)
        JuMP.@constraint(model, [X_vec...] .* vec_to_svec!(ones(cone_dim)) in Hypatia.DoublyNonnegativeTriCone{T}(cone_dim))
    else
        JuMP.@constraint(model, [X_vec...] in MOI.PositiveSemidefiniteConeTriangle(side))
        JuMP.@constraint(model, [X_vec...] .>= 0)
    end

    return model
end

instances[StabilityNumber]["minimal"] = [
    ((2, true),),
    ((2, false),),
    ]
instances[StabilityNumber]["fast"] = [
    ((20, true),),
    ((20, false),),
    ((50, true),),
    ((50, false),),
    ]
instances[StabilityNumber]["slow"] = [
    ((500, true),),
    ((500, false),),
    ]
