#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

let E be a symmetric matrix sparsity pattern:
(1) find sparse PSD matrix with given sparsity pattern, "nearest" to A, fixing some indices F to B
    max_X tr(A, X) :
    tr(X) = 1
    X in intersect(S_+^n, S^n(E))
    X_k = B_k, k in F
(2) find sparse PSD-completable matrix with given sparsity pattern, "nearest" to A, fixing some indices F to B
    max_X tr(A, X) :
    tr(X) = 1
    X in proj_E(S_+^n)
    X_k = B_k, k in F
modified from "Decomposition methods for sparse matrix nearness problems" (2015) by Sun & Vandenberghe
=#

using Test
using LinearAlgebra
using SparseArrays
import Random
import JuMP
const MOI = JuMP.MOI
import Hypatia

# TODO maybe optionally fix some particular values in matrix
function nearestpsdJuMP(
    use_completable::Bool,
    side::Int;
    sparsity::Float64 = sqrt(inv(side)),
    )
    # generate random symmetric A (indefinite) with sparsity pattern E (nonchordal, with diagonal)
    A = tril!(sprandn(side, side, sparsity)) + Diagonal(randn(side))
    (row_idxs, col_idxs, A_vals) = findnz(A)
    num_nz = length(row_idxs)
    @show side
    @show num_nz

    # TODO generate random sample of indices to fix to known values
    # TODO fraction option?

    # setup model
    model = JuMP.Model()
    JuMP.@variable(model, X[1:num_nz])
    JuMP.@objective(model, Max, dot(A_vals, X)) # tr(A, X)
    JuMP.@constraint(model, sum(X[k] for k in eachindex(X) if row_idxs[k] == col_idxs[k]) == 1) # tr(X) == 1
    JuMP.@constraint(model, X in Hypatia.PosSemidefTriSparseCone{Float64, Float64}(side, row_idxs, col_idxs, use_completable))

    return (model = model,)
end

nearestpsdJuMP1() = nearestpsdJuMP(false, 1)
nearestpsdJuMP2() = nearestpsdJuMP(true, 1)
nearestpsdJuMP3() = nearestpsdJuMP(false, 2)
nearestpsdJuMP4() = nearestpsdJuMP(true, 2)
nearestpsdJuMP5() = nearestpsdJuMP(false, 5)
nearestpsdJuMP6() = nearestpsdJuMP(true, 5)
nearestpsdJuMP7() = nearestpsdJuMP(false, 10)
nearestpsdJuMP8() = nearestpsdJuMP(true, 10)
nearestpsdJuMP9() = nearestpsdJuMP(false, 15)
nearestpsdJuMP10() = nearestpsdJuMP(true, 15)
nearestpsdJuMP11() = nearestpsdJuMP(false, 25)
nearestpsdJuMP12() = nearestpsdJuMP(true, 25)
nearestpsdJuMP13() = nearestpsdJuMP(false, 50)
nearestpsdJuMP14() = nearestpsdJuMP(true, 50)
nearestpsdJuMP15() = nearestpsdJuMP(false, 100)
nearestpsdJuMP16() = nearestpsdJuMP(true, 100)
nearestpsdJuMP17() = nearestpsdJuMP(false, 250)
nearestpsdJuMP18() = nearestpsdJuMP(true, 250)
nearestpsdJuMP19() = nearestpsdJuMP(false, 500)
nearestpsdJuMP20() = nearestpsdJuMP(true, 500)
nearestpsdJuMP21() = nearestpsdJuMP(false, 1000)
nearestpsdJuMP22() = nearestpsdJuMP(true, 1000)

function test_nearestpsdJuMP(instance::Function; options, rseed::Int = 1)
    Random.seed!(rseed)
    d = instance()
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return
end

test_nearestpsdJuMP_all(; options...) = test_nearestpsdJuMP.([
    nearestpsdJuMP1,
    nearestpsdJuMP2,
    nearestpsdJuMP3,
    nearestpsdJuMP4,
    nearestpsdJuMP5,
    nearestpsdJuMP6,
    nearestpsdJuMP7,
    nearestpsdJuMP8,
    nearestpsdJuMP9,
    nearestpsdJuMP10,
    nearestpsdJuMP11,
    nearestpsdJuMP12,
    nearestpsdJuMP13,
    nearestpsdJuMP14,
    nearestpsdJuMP15,
    nearestpsdJuMP16,
    nearestpsdJuMP17,
    nearestpsdJuMP18,
    nearestpsdJuMP19,
    nearestpsdJuMP20,
    nearestpsdJuMP21,
    nearestpsdJuMP22,
    ], options = options)

test_nearestpsdJuMP(; options...) = test_nearestpsdJuMP.([
    nearestpsdJuMP1,
    nearestpsdJuMP2,
    nearestpsdJuMP3,
    nearestpsdJuMP4,
    nearestpsdJuMP5,
    nearestpsdJuMP6,
    nearestpsdJuMP11,
    nearestpsdJuMP12,
    nearestpsdJuMP13,
    nearestpsdJuMP14,
    # nearestpsdJuMP15,
    # nearestpsdJuMP16,
    ], options = options)
