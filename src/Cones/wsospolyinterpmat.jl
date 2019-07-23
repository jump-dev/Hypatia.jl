#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial matrix cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

mutable struct WSOSPolyInterpMat{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    point::AbstractVector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    rt2::T
    rt2i::T
    slice::Vector{T}
    LL::Vector{Symmetric{T, Matrix{T}}}
    ΛFs::Vector
    LUs::Vector{Matrix{T}}
    UU1::Matrix{T}
    UU2::Matrix{T}
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact

    function WSOSPolyInterpMat{T}(R::Int, U::Int, Ps::Vector{Matrix{T}}, is_dual::Bool) where {T <: HypReal}
        for Pj in Ps
            @assert size(Pj, 1) == U
        end
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        dim = U * div(R * (R + 1), 2)
        cone.dim = dim
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        return cone
    end
end

WSOSPolyInterpMat{T}(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: HypReal} = WSOSPolyInterpMat{T}(R, U, Ps, false)

function setup_data(cone::WSOSPolyInterpMat{T}) where {T <: HypReal}
    reset_data(cone)
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    cone.grad = Vector{T}(undef, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.rt2 = sqrt(T(2))
    cone.rt2i = inv(cone.rt2)
    cone.slice = Vector{T}(undef, U)
    cone.LL = [Symmetric(zeros(T, size(Pj, 2) * R, size(Pj, 2) * R), :L) for Pj in Ps]
    cone.ΛFs = Vector{Any}(undef, length(Ps))
    cone.LUs = [Matrix{T}(undef, size(Pj, 2), U) for Pj in Ps]
    cone.UU1 = Matrix{T}(undef, U, U)
    cone.UU2 = similar(cone.UU1)
    return
end

get_nu(cone::WSOSPolyInterpMat) = cone.R * sum(size(Pj, 2) for Pj in cone.Ps)

function set_initial_point(arr::AbstractVector, cone::WSOSPolyInterpMat)
    idx = 1
    for i in 1:cone.R, j in 1:i
        arr[idx:(idx + cone.U - 1)] .= (i == j) ? 1 : 0
        idx += cone.U
    end
    return arr
end

_blockrange(inner::Int, outer::Int) = (outer * (inner - 1) + 1):(outer * inner)

function update_feas(cone::WSOSPolyInterpMat)
    @assert !cone.feas_updated
    cone.is_feas = true
    for j in eachindex(cone.Ps)
        Pj = cone.Ps[j]
        LU = cone.LUs[j]
        L = size(Pj, 2)
        Λ = cone.LL[j]
        uo = rowo = 1
        for p in 1:cone.R
            colo = 1
            for q in 1:p
                fact = (p == q ? 1 : cone.rt2i)
                cone.slice .= view(cone.point, uo:(uo + cone.U - 1)) * fact
                mul!(LU, Pj', Diagonal(cone.slice))
                mul!(view(Λ.data, rowo:(rowo + L - 1), colo:(colo + L - 1)), LU, Pj)
                uo += cone.U
                colo += L
            end
            rowo += L
        end
        cone.ΛFs[j] = hyp_chol!(Λ)
        if !isposdef(cone.ΛFs[j])
            cone.is_feas = false
            break
        end
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSPolyInterpMat)
    @assert is_feas(cone)
    cone.grad .= 0
    for j in eachindex(cone.Ps)
        W_inv_j = inv(cone.ΛFs[j]) # TODO store
        Pj = cone.Ps[j]
        L = size(Pj, 2)
        idx = rowo = 1
        for p in 1:cone.R
            colo = 1
            for q in 1:p
                fact = (p == q) ? 1 : cone.rt2
                for i in 1:cone.U
                    cone.grad[idx] -= Pj[i, :]' * view(W_inv_j, rowo:(rowo + L - 1), colo:(colo + L - 1)) * Pj[i, :] * fact
                    idx += 1
                end
                colo += L
            end
            rowo += L
        end
    end
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSPolyInterpMat)
    @assert is_feas(cone)
    cone.hess .= 0
    UU1 = cone.UU1
    UU2 = cone.UU2
    for j in eachindex(cone.Ps)
        W_inv_j = inv(cone.ΛFs[j]) # TODO store
        Pj = cone.Ps[j]
        LU = cone.LUs[j]
        L = size(Pj, 2)
        uo = 0
        for p in 1:cone.R, q in 1:p
            uo += 1
            fact = (p == q) ? 1 : cone.rt2
            rinds = _blockrange(p, L)
            cinds = _blockrange(q, L)
            idxs = _blockrange(uo, cone.U)

            uo2 = 0
            for p2 in 1:cone.R, q2 in 1:p2
                uo2 += 1
                if uo2 < uo
                    continue
                end

                rinds2 = _blockrange(p2, L)
                cinds2 = _blockrange(q2, L)
                idxs2 = _blockrange(uo2, cone.U)

                mul!(LU, view(W_inv_j, rinds, rinds2), Pj')
                mul!(UU1, Pj, LU)
                mul!(LU, view(W_inv_j, cinds, cinds2), Pj')
                mul!(UU2, Pj, LU)
                fact = xor(p == q, p2 == q2) ? cone.rt2i : 1
                @. cone.hess.data[idxs, idxs2] += UU1 * UU2 * fact

                if (p != q) || (p2 != q2)
                    mul!(LU, view(W_inv_j, rinds, cinds2), Pj')
                    mul!(UU1, Pj, LU)
                    mul!(LU, view(W_inv_j, cinds, rinds2), Pj')
                    mul!(UU2, Pj, LU)
                    @. cone.hess.data[idxs, idxs2] += UU1 * UU2 * fact
                end
            end
        end
    end
    cone.hess_updated = true
    return cone.hess
end
