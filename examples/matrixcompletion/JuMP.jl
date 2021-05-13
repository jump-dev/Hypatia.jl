#=
see description in native.jl
=#

using SparseArrays

struct MatrixCompletionJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    size_ratio::Int
    num_rows::Int
end

function build(inst::MatrixCompletionJuMP{T}) where {T <: Float64}
    (size_ratio, num_rows) = (inst.size_ratio, inst.num_rows)
    @assert size_ratio >= 1
    num_cols = size_ratio * num_rows

    (rows, cols, Avals) = findnz(sprandn(num_rows, num_cols, 0.8))
    is_known = vec(Matrix(sparse(rows, cols,
        trues(length(Avals)), num_rows, num_cols)))

    model = JuMP.Model()
    JuMP.@variable(model, X[1:length(is_known)])
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)
    JuMP.@constraint(model, vcat(t, X) in
        MOI.NormSpectralCone(num_rows, num_cols))
    X_unknown = X[.!is_known]
    JuMP.@constraint(model, vcat(1, X_unknown) in
        MOI.GeometricMeanCone(1 + length(X_unknown)))
    JuMP.@constraint(model, X[is_known] .== Avals)

    return model
end
