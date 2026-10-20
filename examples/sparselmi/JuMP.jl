#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/jump-dev/Hypatia.jl
=#

#=
solve a simple LMI problem

min y
st. sum(x) = 1
    Qₖ + y I + sumᵢ xᵢ Pₖᵢ ⪰ 0, ∀k = 1..K
where Qₖ ≺ 0, Pₖᵢ ∈ 𝕊ˢ

formulations: PSD, sparse PSD (sparse Pₖᵢ, with sparse or dense chol),
LMI (sparse or dense Pₖᵢ)
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
    use_cholmod_impl::Bool # use sparsePSD cone with CHOLMOD, else dense impl
end

function build(inst::SparseLMIJuMP{T}) where {T <: Float64}
    @assert +(inst.use_psd, inst.use_sparsepsd, inst.use_linmatrixineq) == 1
    (num_lmis, side_Ps, num_Ps) = (inst.num_lmis, inst.side_Ps, inst.num_Ps)

    function rand_symm()
        if inst.sparse_Ps
            sparsity = min(3.0 / side_Ps, 1.0) # sparsity factor
            M = sprandn(side_Ps, side_Ps, sparsity)
            for idx in rand(1:side_Ps, div(side_Ps, 3))
                M[idx, idx] = rand()
            end
            return M + M'
        else
            M = randn(side_Ps, side_Ps)
            return M + M'
        end
    end
    rand_psd() = (M = rand_symm(); M * M')

    Ps = [rand_symm() for k in 1:num_lmis, i in 1:num_Ps]
    Qs = [-rand_psd() for k in 1:num_lmis]
    matI = one(Qs[1])

    model = JuMP.Model()
    JuMP.@variable(model, y)
    JuMP.@objective(model, Min, y)
    JuMP.@variable(model, x[1:num_Ps])
    JuMP.@constraint(model, sum(x) == 1)
    Ss = [Qs[k] + y * matI + sum(x[i] * Ps[k, i] for i in 1:num_Ps) for k in 1:num_lmis]
    model.ext[:S_vars] = Ss

    if inst.use_psd || inst.use_sparsepsd
        for k in 1:num_lmis
            Sk = Symmetric(Ss[k])
            if inst.use_psd
                JuMP.@constraint(model, Sk in JuMP.PSDCone())
            else
                impl =
                    (inst.use_cholmod_impl ? Cones.PSDSparseCholmod : Cones.PSDSparseDense)
                cone = Hypatia.PosSemidefTriSparseCone{impl, T, T}
                (row_idxs, col_idxs, vals) = findnz(sparse(LowerTriangular(Sk)))
                rt2 = sqrt(2)
                vals_scal = [
                    vals[j] * (row_idxs[j] == col_idxs[j] ? 1.0 : rt2) for
                    j in eachindex(vals)
                ]
                JuMP.@constraint(
                    model,
                    vals_scal in cone(side_Ps, row_idxs, col_idxs, false)
                )
            end
        end
    elseif inst.use_linmatrixineq
        JuMP.@constraint(
            model,
            [k in 1:num_lmis],
            vcat(y, x, 1) in Hypatia.LinMatrixIneqCone{T, T}([matI, Ps[k, :]..., Qs[k]])
        )
    else
        error()
    end

    return model
end

function test_extra(inst::SparseLMIJuMP{T}, model::JuMP.Model) where {T}
    stat = JuMP.termination_status(model)
    @test stat == MOI.OPTIMAL
    (stat == MOI.OPTIMAL) || return

    # the smallest eigenvalue over the LMI matrices is zero at the optimum
    tol = eps(T)^0.2
    λ_min = minimum(eigmin(Symmetric(Matrix(JuMP.value.(S)))) for S in model.ext[:S_vars])
    @test λ_min ≈ 0 atol = tol
    return
end
