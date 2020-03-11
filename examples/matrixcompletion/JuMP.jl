#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

include(joinpath(@__DIR__, "../common_JuMP.jl"))
using SparseArrays

struct MatrixCompletionJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_rows::Int
    num_cols::Int
    nuclearnorm_obj::Bool # use nuclearnorm in the objective, else spectral norm
end

options = ()
example_tests(::Type{MatrixCompletionJuMP{Float64}}, ::MinimalInstances) = [
    ((2, 3, true), false, options),
    ((2, 3, false), false, options),
    ]
example_tests(::Type{MatrixCompletionJuMP{Float64}}, ::FastInstances) = [
    ((5, 8, true), false, options),
    ((5, 8, false), false, options),
    ((12, 24, true), false, options),
    ((12, 24, false), false, options),
    ((14, 140, false), false, options),
    ]
example_tests(::Type{MatrixCompletionJuMP{Float64}}, ::SlowInstances) = [
    ((14, 140, true), false, options),
    ((40, 70, true), false, options),
    ((40, 70, false), false, options),
    ((18, 180, true), false, options),
    ((18, 180, false), false, options),
    ]

function build(inst::MatrixCompletionJuMP{T}) where {T <: Float64} # TODO generic reals
    (num_rows, num_cols) = (inst.num_rows, inst.num_cols)
    @assert num_rows <= num_cols
    A = randn(num_rows, num_cols)
    (row_idxs, col_idxs, _) = findnz(sprand(Bool, num_rows, num_cols, 0.1))

    model = JuMP.Model()
    JuMP.@variable(model, X[1:num_rows, 1:num_cols])
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)
    cone = (inst.nuclearnorm_obj ? MOI.NormNuclearCone : MOI.NormSpectralCone)
    JuMP.@constraint(model, vcat(t, vec(X)) in cone(num_rows, num_cols))
    JuMP.@constraint(model, [(row, col) in zip(row_idxs, col_idxs)], X[row, col] == A[row, col])

    return model
end

function test_extra(inst::MatrixCompletionJuMP, model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

# @testset "MatrixCompletionJuMP" for inst in example_tests(MatrixCompletionJuMP{Float64}, MinimalInstances()) test(inst...) end

return MatrixCompletionJuMP
