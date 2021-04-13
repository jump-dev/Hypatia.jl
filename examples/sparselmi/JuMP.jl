#=
solve a simple LMI problem

min c⋅x
st. x₁ = 0
    sumᵢ Pₖ₀ + xᵢ Pₖᵢ ⪰ 0, ∀k = 1..K
where Pₖᵢ ∈ 𝕊ˢ, Pₖ₀ ≺ 0, Pₖ₁ = I, Pₖᵢ ⪰ 0 ∀k > 0

formulations: PSD, sparse PSD (sparse Pₖᵢ, with sparse or dense chol), LMI (sparse or dense Pₖᵢ)
=#

using SparseArrays

struct SparseLMIJuMP{T <: Real} <: ExampleInstanceJuMP{T}
    num_lmis::Int
    side_Ps::Int
    num_Ps::Int
    sparse_Ps::Bool
    use_psd::Bool # use PSD cone
    use_linmatrixineq::Bool # use linmatrixineq cone
    use_sparsepsd::Bool # use sparsePSD cone
    use_cholmod_impl::Bool # use sparsePSD cone with CHOLMOD implementation, else dense implementation
end

function build(inst::SparseLMIJuMP{T}) where {T <: Float64}
    @assert (inst.use_psd + inst.use_sparsepsd + inst.use_linmatrixineq) == 1
    (num_lmis, side_Ps, num_Ps) = (inst.num_lmis, inst.side_Ps, inst.num_Ps)

    # TODO make sparse if inst.sparse_Ps
    rand_psd() = (M = randn(side_Ps, side_Ps); Symmetric(M * M'))

    P0s = Vector(undef, num_lmis)
    Ps = Matrix(undef, num_lmis, num_Ps)
    for k in 1:num_lmis
        for i in 2:num_Ps
            Ps[k, i] = rand_psd() # ≻ 0
        end
        P0s[k] = -sum(Ps[k, 2:end]) # ≺ 0, feasible for x = 1
        # TODO Ps[k, 1] = I
        Ps[k, 1] = Symmetric(one(P0s[k]))
    end
    c = rand(num_Ps)

    model = JuMP.Model()
    JuMP.@variable(model, x[1:num_Ps])
    JuMP.@objective(model, Min, dot(c, x))
    JuMP.@constraint(model, x[1] == 0)

    if inst.use_psd || inst.use_sparsepsd
        for k in 1:num_lmis
            Sk = Symmetric(P0s[k] + sum(x[i] * Ps[k, i] for i in 1:num_Ps))
            if inst.use_psd
                JuMP.@constraint(model, Sk in JuMP.PSDCone())
            else
                impl = (inst.use_cholmod_impl ? Cones.PSDSparseCholmod : Cones.PSDSparseDense)
                cone = Hypatia.PosSemidefTriSparseCone{impl, Float64, Float64}
                (row_idxs, col_idxs, vals) = findnz(sparse(LowerTriangular(Sk)))
                JuMP.@constraint(model, vals in cone(side_Ps, row_idxs, col_idxs, false))
            end
        end
    elseif inst.use_linmatrixineq
        JuMP.@constraint(model, [k in 1:num_lmis],
            vcat(x, 1) in Hypatia.LinMatrixIneqCone{Float64}([Ps[k, :]..., P0s[k]]))
    else
        error()
    end

    return model
end
