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

function nearestpsd_JuMP(
    ::Type{T},
    side::Int,
    use_completable::Bool, # solve problem (2) above, else solve problem (1)
    use_chordal_sparsity::Bool, # use a chordal sparsity pattern, else use a general sparsity pattern
    use_sparsepsd::Bool = true; # use sparse PSD cone formulation, else dense PSD formulation
    sparsity::Float64 = min(3 / side, 1.0), # sparsity factor (before computing optional chordal extension)
    ) where {T <: Float64} # TODO support generic reals
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

    return (model, ())
end

function test_nearestpsd_JuMP(instance::Tuple; T::Type{<:Real} = Float64, options::NamedTuple = NamedTuple(), rseed::Int = 1)
    Random.seed!(rseed)
    d = nearestpsd_JuMP(T, instance...)
    JuMP.set_optimizer(d.model, () -> Hypatia.Optimizer{T}(; options...))
    JuMP.optimize!(d.model)
    @test JuMP.termination_status(d.model) == MOI.OPTIMAL
    return d.model.moi_backend.optimizer.model.optimizer.result
end

nearestpsd_JuMP_fast = [
    (1, false, true, true),
    (1, false, false, true),
    (1, true, true, true),
    (1, true, false, true),
    (1, false, true, false),
    (1, false, false, false),
    (1, true, true, false),
    (1, true, false, false),
    (5, false, true, true),
    (5, false, false, true),
    (5, true, true, true),
    (5, true, false, true),
    (5, false, true, false),
    (5, false, false, false),
    (5, true, true, false),
    (5, true, false, false),
    (20, false, true, true),
    (20, false, false, true),
    (20, true, true, true),
    (20, true, false, true),
    (20, false, true, false),
    (20, false, false, false),
    (20, true, true, false),
    (20, true, false, false),
    (100, false, true, false),
    (100, false, false, false),
    ]
nearestpsd_JuMP_slow = [
    (100, false, true, true),
    (100, false, false, true),
    (100, true, true, true),
    (100, true, false, true),
    (100, true, true, false),
    (100, true, false, false),
    ]
