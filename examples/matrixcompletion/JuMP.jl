#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

using SparseArrays

struct MatrixCompletionJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_rows::Int
    num_cols::Int
end

function build(inst::MatrixCompletionJuMP{T}) where {T <: Float64} # TODO generic reals
    (num_rows, num_cols) = (inst.num_rows, inst.num_cols)
    @assert num_rows <= num_cols
    (rows, cols, Avals) = findnz(sprand(num_rows, num_cols, 0.1))
    mat_to_vec_idx(i::Int, j::Int) = (j - 1) * num_rows + i
    is_unknown = fill(true, num_rows * num_cols)
    for (i, j) in zip(rows, cols)
        is_unknown[mat_to_vec_idx(i, j)] = false
    end

    model = JuMP.Model()
    JuMP.@variable(model, X[1:num_rows, 1:num_cols])
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)
    JuMP.@constraint(model, vcat(t, vec(X)) in MOI.NormSpectralCone(num_rows, num_cols))
    JuMP.@constraint(model, vcat(1, X[is_unknown]) in MOI.GeometricMeanCone(1 + sum(is_unknown)))
    JuMP.@constraint(model, X[.!is_unknown] .== Avals)

    return model
end
