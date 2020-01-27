#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

let E be a symmetric matrix sparsity pattern:
(1) find sparse PSD matrix with given sparsity pattern, "nearest" to A
    max_X tr(A, X) :
    tr(X) = 1
    X in intersect(S_+^n, S^n(E))
(2) find sparse PSD-completable matrix with given sparsity pattern, "nearest" to A
    max_X tr(A, X) :
    tr(X) = 1
    X in proj_E(S_+^n)

adapted from "Decomposition methods for sparse matrix nearness problems" (2015) by Sun & Vandenberghe
=#

using Test
using LinearAlgebra
using SparseArrays
import Random
import JuMP
const MOI = JuMP.MOI
import Hypatia
import Hypatia.Cones

function nearestpsdJuMP(
    use_completable::Bool,
    use_chordal_sparsity::Bool,
    side::Int;
    sparsity::Float64 = min(3 / side, 1.0),
    )
    # generate random symmetric A (indefinite) with sparsity pattern E (nonchordal, with diagonal)
    A = tril!(sprandn(side, side, sparsity)) + Diagonal(randn(side))
    if use_chordal_sparsity
        # compute a chordal extension of A using CHOLMOD functions
        # TODO extend ModelUtilities to compute chordal extensions
        copyto!(A, I)
        A = sparse(cholesky(Symmetric(A, :L)).L)
        (row_idxs, col_idxs, _) = findnz(A)
        A_vals = randn(length(row_idxs))
    else
        (row_idxs, col_idxs, A_vals) = findnz(A)
    end
    num_nz = length(row_idxs)

    model = JuMP.Model()
    JuMP.@variable(model, X[1:num_nz])
    JuMP.@objective(model, Max, dot(A_vals, X)) # tr(A, X)
    JuMP.@constraint(model, sum(X[k] for k in eachindex(X) if row_idxs[k] == col_idxs[k]) == 1) # tr(X) == 1
    JuMP.@constraint(model, X in Hypatia.PosSemidefTriSparseCone{Float64, Float64}(side, row_idxs, col_idxs, use_completable))

    return (model = model,)
end

nearestpsdJuMP1() = nearestpsdJuMP(false, true, 1)
nearestpsdJuMP2() = nearestpsdJuMP(true, true, 1)
nearestpsdJuMP3() = nearestpsdJuMP(false, false, 1)
nearestpsdJuMP4() = nearestpsdJuMP(true, false, 1)
nearestpsdJuMP5() = nearestpsdJuMP(false, true, 5)
nearestpsdJuMP6() = nearestpsdJuMP(true, true, 5)
nearestpsdJuMP7() = nearestpsdJuMP(false, false, 5)
nearestpsdJuMP8() = nearestpsdJuMP(true, false, 5)
nearestpsdJuMP9() = nearestpsdJuMP(false, true, 20)
nearestpsdJuMP10() = nearestpsdJuMP(true, true, 20)
nearestpsdJuMP11() = nearestpsdJuMP(false, false, 20)
nearestpsdJuMP12() = nearestpsdJuMP(true, false, 20)
nearestpsdJuMP13() = nearestpsdJuMP(false, true, 50)
nearestpsdJuMP14() = nearestpsdJuMP(true, true, 50)
nearestpsdJuMP15() = nearestpsdJuMP(false, false, 50)
nearestpsdJuMP16() = nearestpsdJuMP(true, false, 50)
nearestpsdJuMP17() = nearestpsdJuMP(false, true, 200)
nearestpsdJuMP18() = nearestpsdJuMP(true, true, 200)
nearestpsdJuMP19() = nearestpsdJuMP(false, false, 200)
nearestpsdJuMP20() = nearestpsdJuMP(true, false, 200)
nearestpsdJuMP21() = nearestpsdJuMP(false, true, 1000)
nearestpsdJuMP22() = nearestpsdJuMP(true, true, 1000)
nearestpsdJuMP23() = nearestpsdJuMP(false, false, 1000)
nearestpsdJuMP24() = nearestpsdJuMP(true, false, 1000)

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
    nearestpsdJuMP23,
    nearestpsdJuMP24,
    ], options = options)

test_nearestpsdJuMP(; options...) = test_nearestpsdJuMP.([
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
    ], options = options)
