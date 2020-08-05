#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

TODO
decide on a different name because it's a strengthening
related problems: Conic approach to quantum graph parameters using linear optimization over the completely positive semidefinite cone M. Laurent, T. Piovesan
add sparse PSD formulation
test

=#

using SparseArrays
import Hypatia.ModelUtilities.vec_to_svec!

struct ThetaNumber{T <: Real} <: ExampleInstanceJuMP{T}
    side::Int
    use_dnn::Bool
end

function build(inst::ThetaNumber{T}) where {T <: Float64} # TODO generic reals
    (side, use_dnn) = (inst.side, inst.use_dnn)

    sparsity = min(3.0 / side, 1.0)
    graph = tril!(sprandn(side, side, sparsity)) + Diagonal(ones(side))
    (row_idxs, col_idxs, A_vals) = findnz(graph)
    diag_idxs = findall(row_idxs .== col_idxs)

    model = JuMP.Model()
    JuMP.@variable(model, X[1:length(row_idxs)])
    JuMP.@objective(model, Max, sum(X))
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

instances[ThetaNumber]["minimal"] = [
    ((1, true),),
    ((1, false),),
    ]
instances[ThetaNumber]["fast"] = [
    ((50, true),),
    ((50, false),),
    ]

instances[ThetaNumber]["slow"] = [
    ((500, true),),
    ((500, false),),
    ]
