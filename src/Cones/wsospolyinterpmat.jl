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
    ipwt::Vector{Matrix{T}}
    point::Vector{T}

    point::AbstractVector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool

    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    tmp_hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact
    mat::Vector{Matrix{T}}
    matfact::Vector
    tmp1::Vector{Matrix{T}}
    tmp2::Matrix{T}
    tmp3::Matrix{T}

    function WSOSPolyInterpMat{T}(R::Int, U::Int, ipwt::Vector{Matrix{T}}, is_dual::Bool) where {T <: Real}
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == U
        end
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        dim = U * div(R * (R + 1), 2)
        cone.dim = dim
        cone.R = R
        cone.U = U
        cone.ipwt = ipwt
        return cone
    end
end

WSOSPolyInterpMat{T}(R::Int, U::Int, ipwt::Vector{Matrix{T}}) where {T <: Real} = WSOSPolyInterpMat{T}(R, U, ipwt, false)

function setup_data(cone::WSOSPolyInterpMat{T}) where {T <: HypReal}
    reset_data(cone)
    dim = cone.dim
    U = cone.U
    R = cone.R
    ipwt = cone.ipwt
    cone.grad = Vector{T}(undef, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.mat = [similar(cone.hess, size(ipwtj, 2) * R, size(ipwtj, 2) * R) for ipwtj in ipwt]
    cone.matfact = Vector{Any}(undef, length(ipwt))
    cone.tmp1 = [similar(cone.hess, size(ipwtj, 2), U) for ipwtj in ipwt]
    cone.tmp2 = similar(cone.hess, U, U)
    cone.tmp3 = similar(cone.tmp2)
    return
end

get_nu(cone::WSOSPolyInterpMat) = cone.R * sum(size(ipwtj, 2) for ipwtj in cone.ipwt)

function set_initial_point(arr::AbstractVector{T}, cone::WSOSPolyInterpMat{T}) where {T <: Real}
    # sum of diagonal matrices with interpolant polynomial repeating on the diagonal
    idx = 1
    for i in 1:cone.R, j in 1:i
        arr[idx:(idx + cone.U - 1)] .= (i == j) ? one(T) : zero(T)
        idx += cone.U
    end
    return arr
end

_blockrange(inner::Int, outer::Int) = (outer * (inner - 1) + 1):(outer * inner)

function update_feas(cone::WSOSPolyInterpMat)
    @assert !cone.feas_updated
    rt2i = inv(sqrt(2))
    cone.is_feas = true
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        L = size(ipwtj, 2)
        mat = cone.mat[j]
        uo = 1
        for p in 1:cone.R, q in 1:p
            point_pq = cone.point[uo:(uo + cone.U - 1)] # TODO prealloc
            if p != q
                @. point_pq *= rt2i
            end

            @. tmp1j = ipwtj' * point_pq'

            rinds = _blockrange(p, L)
            cinds = _blockrange(q, L)
            mul!(view(mat, rinds, cinds), tmp1j, ipwtj)

            uo += cone.U
        end

        cone.matfact[j] = cholesky!(Symmetric(mat, :L), check = false)
        if !isposdef(cone.matfact[j])
            cone.is_feas = false
            break
        end
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSPolyInterpMat)
    @assert cone.is_feas
    rt2 = sqrt(2)
    cone.grad .= 0
    for j in eachindex(cone.ipwt)
        W_inv_j = inv(cone.matfact[j])

        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        tmp2 = cone.tmp2
        tmp3 = cone.tmp3

        L = size(ipwtj, 2)
        uo = 0
        for p in 1:cone.R, q in 1:p
            uo += 1
            fact = (p == q) ? 1 : rt2
            rinds = _blockrange(p, L)
            cinds = _blockrange(q, L)
            idxs = _blockrange(uo, cone.U)

            for i in 1:cone.U
                cone.grad[idxs[i]] -= ipwtj[i, :]' * view(W_inv_j, rinds, cinds) * ipwtj[i, :] * fact
            end
        end
    end
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSPolyInterpMat)
    rt2 = sqrt(2)
    rt2i = inv(rt2)
    cone.hess .= 0
    for j in eachindex(cone.ipwt)
        W_inv_j = inv(cone.matfact[j]) # TODO store

        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        tmp2 = cone.tmp2
        tmp3 = cone.tmp3

        L = size(ipwtj, 2)
        uo = 0
        for p in 1:cone.R, q in 1:p
            uo += 1
            fact = (p == q) ? 1 : rt2
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

                mul!(tmp1j, view(W_inv_j, rinds, rinds2), ipwtj')
                mul!(tmp2, ipwtj, tmp1j)
                mul!(tmp1j, view(W_inv_j, cinds, cinds2), ipwtj')
                mul!(tmp3, ipwtj, tmp1j)
                fact = xor(p == q, p2 == q2) ? rt2i : 1
                @. cone.hess.data[idxs, idxs2] += tmp2 * tmp3 * fact

                if (p != q) || (p2 != q2)
                    mul!(tmp1j, view(W_inv_j, rinds, cinds2), ipwtj')
                    mul!(tmp2, ipwtj, tmp1j)
                    mul!(tmp1j, view(W_inv_j, cinds, rinds2), ipwtj')
                    mul!(tmp3, ipwtj, tmp1j)
                    @. cone.hess.data[idxs, idxs2] += tmp2 * tmp3 * fact
                end
            end
        end
    end
    cone.hess_updated = true
    return cone.hess
end

# function check_in_cone(cone::WSOSPolyInterpMat{T}) where {T <: HypReal}
#     rt2 = sqrt(T(2))
#     rt2i = inv(rt2)
#
#     for j in eachindex(cone.ipwt)
#         ipwtj = cone.ipwt[j]
#         tmp1j = cone.tmp1[j]
#         L = size(ipwtj, 2)
#         mat = cone.mat[j]
#
#         uo = 1
#         for p in 1:cone.R, q in 1:p
#             point_pq = cone.point[uo:(uo + cone.U - 1)] # TODO prealloc
#             if p != q
#                 @. point_pq *= rt2i
#             end
#             @. tmp1j = ipwtj' * point_pq'
#
#             rinds = _blockrange(p, L)
#             cinds = _blockrange(q, L)
#             mul!(view(mat, rinds, cinds), tmp1j, ipwtj)
#
#             uo += cone.U
#         end
#
#         cone.matfact[j] = hyp_chol!(Symmetric(mat, :L))
#         if !isposdef(cone.matfact[j])
#             return false
#         end
#     end
#
#     cone.grad .= zero(T)
#     cone.hess .= zero(T)
#     for j in eachindex(cone.ipwt)
#         W_inv_j = inv(cone.matfact[j])
#
#         ipwtj = cone.ipwt[j]
#         tmp1j = cone.tmp1[j]
#         tmp2 = cone.tmp2
#         tmp3 = cone.tmp3
#
#         L = size(ipwtj, 2)
#         uo = 0
#         for p in 1:cone.R, q in 1:p
#             uo += 1
#             fact = (p == q) ? one(T) : rt2
#             rinds = _blockrange(p, L)
#             cinds = _blockrange(q, L)
#             idxs = _blockrange(uo, cone.U)
#
#             for i in 1:cone.U
#                 cone.grad[idxs[i]] -= ipwtj[i, :]' * view(W_inv_j, rinds, cinds) * ipwtj[i, :] * fact
#             end
#
#             uo2 = 0
#             for p2 in 1:cone.R, q2 in 1:p2
#                 uo2 += 1
#                 if uo2 < uo
#                     continue
#                 end
#
#                 rinds2 = _blockrange(p2, L)
#                 cinds2 = _blockrange(q2, L)
#                 idxs2 = _blockrange(uo2, cone.U)
#
#                 mul!(tmp1j, view(W_inv_j, rinds, rinds2), ipwtj')
#                 mul!(tmp2, ipwtj, tmp1j)
#                 mul!(tmp1j, view(W_inv_j, cinds, cinds2), ipwtj')
#                 mul!(tmp3, ipwtj, tmp1j)
#                 fact = xor(p == q, p2 == q2) ? rt2i : one(T)
#                 @. cone.hess[idxs, idxs2] += tmp2 * tmp3 * fact
#
#                 if (p != q) || (p2 != q2)
#                     mul!(tmp1j, view(W_inv_j, rinds, cinds2), ipwtj')
#                     mul!(tmp2, ipwtj, tmp1j)
#                     mul!(tmp1j, view(W_inv_j, cinds, rinds2), ipwtj')
#                     mul!(tmp3, ipwtj, tmp1j)
#                     @. cone.hess[idxs, idxs2] += tmp2 * tmp3 * fact
#                 end
#             end
#         end
#     end
#
#     return factorize_hess(cone)
# end
