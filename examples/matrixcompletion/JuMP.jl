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

function matrixcompletionJuMP(
    num_rows::Int,
    num_cols::Int;
    nuclearnorm_obj::Bool = true, # use nuclearnorm in the objective, else spectral norm
    )
    @assert num_rows <= num_cols
    A = randn(num_rows, num_cols)
    (row_idxs, col_idxs, _) = findnz(sprand(Bool, num_rows, num_cols, 0.1))

    model = JuMP.Model()
    JuMP.@variable(model, X[1:num_rows, 1:num_cols])
    JuMP.@variable(model, t)
    JuMP.@objective(model, Min, t)
    cone = (nuclearnorm_obj ? MOI.NormNuclearCone(num_rows, num_cols) : MOI.NormSpectralCone(num_rows, num_cols))
    JuMP.@constraint(model, vcat(t, vec(X)) in cone)
    JuMP.@constraint(model, [(row, col) in zip(row_idxs, col_idxs)], X[row, col] == A[row, col])

    return (model = model,)
end

matrixcompletionJuMP1() = matrixcompletionJuMP(5, 8, nuclearnorm_obj = true)
matrixcompletionJuMP2() = matrixcompletionJuMP(5, 8, nuclearnorm_obj = false)
matrixcompletionJuMP3() = matrixcompletionJuMP(15, 20, nuclearnorm_obj = true)
matrixcompletionJuMP4() = matrixcompletionJuMP(15, 20, nuclearnorm_obj = false)
matrixcompletionJuMP5() = matrixcompletionJuMP(40, 70, nuclearnorm_obj = true)
matrixcompletionJuMP6() = matrixcompletionJuMP(40, 70, nuclearnorm_obj = false)

function test_matrixcompletionJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_matrixcompletionJuMP_all(; options...) = test_matrixcompletionJuMP.([
    matrixcompletionJuMP1,
    matrixcompletionJuMP2,
    matrixcompletionJuMP3,
    matrixcompletionJuMP4,
    matrixcompletionJuMP5,
    matrixcompletionJuMP6,
    ], options = options)

test_matrixcompletionJuMP(; options...) = test_matrixcompletionJuMP.([
    matrixcompletionJuMP1,
    matrixcompletionJuMP2,
    matrixcompletionJuMP3,
    matrixcompletionJuMP4,
    ], options = options)
