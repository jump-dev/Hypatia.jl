#=
let E be a symmetric matrix sparsity pattern:
(1) find sparse PSD matrix with given sparsity pattern, "nearest" to A
    max_X tr(A, X) :
    tr(X) = 1
    X in intersect(S_+^n, S^n(E))
(2) find sparse PSD-completable matrix with given sparsity pattern, "nearest" to A
    max_X tr(A, X) :
    tr(X) = 1
    X in proj_E(S_+^n)

adapted from
"Decomposition methods for sparse matrix nearness problems"
(2015) by Sun & Vandenberghe
=#

using SparseArrays

struct NearestPSDJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    side::Int
    use_completable::Bool # solve problem (2) above, else solve problem (1)
    use_chordal_sparsity::Bool # use a chordal sparsity pattern
    use_sparsepsd::Bool # use sparse PSD cone formulation, else dense PSD cone
    use_cholmod_impl::Bool # use sparsePSD CHOLMOD implementation, else dense
end

function build(inst::NearestPSDJuMP{T}) where {T <: Float64}
    side = inst.side
    sparsity = min(3.0 / side, 1.0) # sparsity factor (before chordal extension)

    # generate random symmetric A (indefinite) with sparsity pattern E (nonchordal)
    A = tril!(sprandn(side, side, sparsity)) + Diagonal(randn(side))
    if inst.use_chordal_sparsity
        # compute a (heuristic) chordal extension of A using CHOLMOD functions
        copyto!(A, I)
        A = sparse(cholesky(Symmetric(A, :L)).L)
        (row_idxs, col_idxs, _) = findnz(A)
        A_vals = randn(length(row_idxs))
    else
        (row_idxs, col_idxs, A_vals) = findnz(A)
    end
    diag_idxs = findall(row_idxs .== col_idxs)

    model = JuMP.Model()

    if inst.use_sparsepsd || !inst.use_completable
        JuMP.@variable(model, X[1:length(row_idxs)])
        JuMP.@objective(model, Max, 2 * dot(A_vals, X) -
            sum(A_vals[k] * X[k] for k in diag_idxs)) # tr(A, X)
        JuMP.@constraint(model, sum(X[diag_idxs]) == 1) # tr(X) == 1

        if inst.use_sparsepsd
            rt2 = sqrt(2)
            X_scal = [X[k] * (row_idxs[k] == col_idxs[k] ? 1.0 : rt2)
                for k in eachindex(X)]
            impl = (inst.use_cholmod_impl ? Cones.PSDSparseCholmod :
                Cones.PSDSparseDense)
            cone = Hypatia.PosSemidefTriSparseCone{impl, T, T}
            JuMP.@constraint(model, X_scal in cone(side, row_idxs,
                col_idxs, inst.use_completable))
        else
            X_sparse = sparse(row_idxs, col_idxs, X)
            JuMP.@SDconstraint(model, Symmetric(Matrix(X_sparse), :L) >= 0)
        end
    else
        @assert inst.use_completable
        JuMP.@variable(model, X[1:side, 1:side], PSD)
        # tr(A, X):
        JuMP.@objective(model, Max, 2 * sum(X[row_idxs[k], col_idxs[k]] * A_vals[k]
            for k in eachindex(row_idxs)) - dot(A_vals[diag_idxs], diag(X)))
        JuMP.@constraint(model, tr(X) == 1) # tr(X) == 1
    end

    return model
end
