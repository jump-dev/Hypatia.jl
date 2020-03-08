#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

see description in native.jl
=#

using LinearAlgebra
import JuMP
const MOI = JuMP.MOI
import Random
using SparseArrays
import Hypatia
const CO = Hypatia.Cones
using Test

function matrixcompletion_JuMP(
    ::Type{T},
    num_rows::Int,
    num_cols::Int,
    nuclearnorm_obj::Bool, # use nuclearnorm in the objective, else spectral norm
    ) where {T <: Float64} # TODO support generic reals
    @assert num_rows <= num_cols
    A = randn(num_rows, num_cols)
    (row_idxs, col_idxs, _) = findnz(sprand(Bool, num_rows, num_cols, 0.1))

    model = JuMP.Model()
    JuMP.@variable(model, X[1:num_rows, 1:num_cols])
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)
    cone = (nuclearnorm_obj ? MOI.NormNuclearCone : MOI.NormSpectralCone)
    JuMP.@constraint(model, vcat(t, vec(X)) in cone(num_rows, num_cols))
    JuMP.@constraint(model, [(row, col) in zip(row_idxs, col_idxs)], X[row, col] == A[row, col])

    return (model = model,)
end

function test_matrixcompletion_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = matrixcompletion_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

matrixcompletion_JuMP_fast = [
    (5, 8, true),
    (5, 8, false),
    (15, 20, true),
    (15, 20, false),
    (14, 140, true),
    (14, 140, false),
    ]
matrixcompletion_JuMP_slow = [
    (40, 70, true),
    (40, 70, false),
    (18, 180, true),
    (18, 180, false),
    ]
