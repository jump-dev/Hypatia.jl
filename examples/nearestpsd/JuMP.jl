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
import Hypatia.ModelUtilities

function nearestpsdJuMP(
    use_completable::Bool, # solve problem (2) above, else solve problem (1)
    use_chordal_sparsity::Bool, # use a chordal sparsity pattern, else use a general sparsity pattern
    side::Int;
    use_sparsepsd::Bool = true, # use sparse PSD cone formulation, else dense PSD formulation
    sparsity::Float64 = min(3 / side, 1.0), # sparsity factor (before computing optional chordal extension)
    )
    # generate random symmetric A (indefinite) with sparsity pattern E (nonchordal, with diagonal)
    A = tril!(sprandn(side, side, sparsity)) + Diagonal(randn(side))
    if use_chordal_sparsity
        # compute a (heuristic) chordal extension of A using CHOLMOD functions
        # TODO extend ModelUtilities to compute chordal extensions
        copyto!(A, I)
        A = sparse(cholesky(Symmetric(A, :L)).L)
        (row_idxs, col_idxs, _) = findnz(A)
        A_vals = randn(length(row_idxs))
    else
        (row_idxs, col_idxs, A_vals) = findnz(A)
    end
    num_nz = length(row_idxs)
    diag_idxs = Int[]
    for k in 1:num_nz
        if row_idxs[k] == col_idxs[k]
            push!(diag_idxs, k)
        end
    end

    model = JuMP.Model()

    if use_sparsepsd || !use_completable
        JuMP.@variable(model, X[1:num_nz])
        JuMP.@objective(model, Max, 2 * dot(A_vals, X) - sum(A_vals[k] * X[k] for k in diag_idxs)) # tr(A, X)
        JuMP.@constraint(model, sum(X[diag_idxs]) == 1) # tr(X) == 1
        if use_sparsepsd
            rt2 = sqrt(2)
            X_scal = [X[k] * (row_idxs[k] == col_idxs[k] ? 1.0 : rt2) for k in eachindex(X)]
            JuMP.@constraint(model, X_scal in Hypatia.PosSemidefTriSparseCone{Float64, Float64}(side, row_idxs, col_idxs, use_completable))
        else
            X_sparse = sparse(row_idxs, col_idxs, X)
            JuMP.@SDconstraint(model, Symmetric(Matrix(X_sparse), :L) >= 0)
        end
    else
        @assert use_completable
        JuMP.@variable(model, X[1:side, 1:side], PSD)
        JuMP.@objective(model, Max, 2 * sum(X[row_idxs[k], col_idxs[k]] * A_vals[k] for k in eachindex(row_idxs)) - dot(A_vals[diag_idxs], diag(X))) # tr(A, X)
        JuMP.@constraint(model, tr(X) == 1) # tr(X) == 1
    end

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
nearestpsdJuMP25() = nearestpsdJuMP(false, true, 1, use_sparsepsd = false)
nearestpsdJuMP26() = nearestpsdJuMP(true, true, 1, use_sparsepsd = false)
nearestpsdJuMP27() = nearestpsdJuMP(false, false, 1, use_sparsepsd = false)
nearestpsdJuMP28() = nearestpsdJuMP(true, false, 1, use_sparsepsd = false)
nearestpsdJuMP29() = nearestpsdJuMP(false, true, 5, use_sparsepsd = false)
nearestpsdJuMP30() = nearestpsdJuMP(true, true, 5, use_sparsepsd = false)
nearestpsdJuMP31() = nearestpsdJuMP(false, false, 5, use_sparsepsd = false)
nearestpsdJuMP32() = nearestpsdJuMP(true, false, 5, use_sparsepsd = false)
nearestpsdJuMP33() = nearestpsdJuMP(false, true, 20, use_sparsepsd = false)
nearestpsdJuMP34() = nearestpsdJuMP(true, true, 20, use_sparsepsd = false)
nearestpsdJuMP35() = nearestpsdJuMP(false, false, 20, use_sparsepsd = false)
nearestpsdJuMP36() = nearestpsdJuMP(true, false, 20, use_sparsepsd = false)
nearestpsdJuMP37() = nearestpsdJuMP(false, true, 50, use_sparsepsd = false)
nearestpsdJuMP38() = nearestpsdJuMP(true, true, 50, use_sparsepsd = false)
nearestpsdJuMP39() = nearestpsdJuMP(false, false, 50, use_sparsepsd = false)
nearestpsdJuMP40() = nearestpsdJuMP(true, false, 50, use_sparsepsd = false)
nearestpsdJuMP41() = nearestpsdJuMP(false, true, 200, use_sparsepsd = false)
nearestpsdJuMP42() = nearestpsdJuMP(true, true, 200, use_sparsepsd = false)
nearestpsdJuMP43() = nearestpsdJuMP(false, false, 200, use_sparsepsd = false)
nearestpsdJuMP44() = nearestpsdJuMP(true, false, 200, use_sparsepsd = false)
nearestpsdJuMP45() = nearestpsdJuMP(false, true, 1000, use_sparsepsd = false)
nearestpsdJuMP46() = nearestpsdJuMP(true, true, 1000, use_sparsepsd = false)
nearestpsdJuMP47() = nearestpsdJuMP(false, false, 1000, use_sparsepsd = false)
nearestpsdJuMP48() = nearestpsdJuMP(true, false, 1000, use_sparsepsd = false)

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
    nearestpsdJuMP25,
    nearestpsdJuMP26,
    nearestpsdJuMP27,
    nearestpsdJuMP28,
    nearestpsdJuMP29,
    nearestpsdJuMP30,
    nearestpsdJuMP31,
    nearestpsdJuMP32,
    nearestpsdJuMP33,
    nearestpsdJuMP34,
    nearestpsdJuMP35,
    nearestpsdJuMP36,
    nearestpsdJuMP37,
    nearestpsdJuMP38,
    nearestpsdJuMP39,
    nearestpsdJuMP40,
    nearestpsdJuMP41,
    nearestpsdJuMP42,
    nearestpsdJuMP43,
    nearestpsdJuMP44,
    nearestpsdJuMP45,
    nearestpsdJuMP46,
    nearestpsdJuMP47,
    nearestpsdJuMP48,
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
    nearestpsdJuMP25,
    nearestpsdJuMP26,
    nearestpsdJuMP27,
    nearestpsdJuMP28,
    nearestpsdJuMP29,
    nearestpsdJuMP30,
    nearestpsdJuMP31,
    nearestpsdJuMP32,
    nearestpsdJuMP33,
    nearestpsdJuMP34,
    nearestpsdJuMP35,
    nearestpsdJuMP36,
    nearestpsdJuMP37,
    # nearestpsdJuMP38,
    nearestpsdJuMP39,
    # nearestpsdJuMP40,
    ], options = options)
