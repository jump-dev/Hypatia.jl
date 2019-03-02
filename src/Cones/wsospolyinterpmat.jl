#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial matrix cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

mutable struct WSOSPolyInterpMat <: Cone
    use_dual::Bool
    dim::Int
    R::Int
    U::Int
    ipwt::Vector{Matrix{Float64}}
    point::Vector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    mat::Vector{Matrix{Float64}}
    matfact::Vector{CholeskyPivoted{Float64, Matrix{Float64}}}
    tmp1::Vector{Matrix{Float64}}
    tmp2::Matrix{Float64}
    tmp3::Matrix{Float64}

    function WSOSPolyInterpMat(R::Int, U::Int, ipwt::Vector{Matrix{Float64}}, is_dual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == U
        end
        cone = new()
        cone.use_dual = !is_dual # using dual barrier
        dim = U * div(R * (R + 1), 2)
        cone.dim = dim
        cone.R = R
        cone.U = U
        cone.ipwt = ipwt
        cone.point = similar(ipwt[1], dim)
        cone.g = similar(ipwt[1], dim)
        cone.H = similar(ipwt[1], dim, dim)
        cone.H2 = similar(cone.H)
        cone.mat = [similar(ipwt[1], size(ipwtj, 2) * R, size(ipwtj, 2) * R) for ipwtj in ipwt]
        cone.matfact = Vector{CholeskyPivoted{Float64, Matrix{Float64}}}(undef, length(ipwt))
        cone.tmp1 = [similar(ipwt[1], size(ipwtj, 2), U) for ipwtj in ipwt]
        cone.tmp2 = similar(ipwt[1], U, U)
        cone.tmp3 = similar(cone.tmp2)
        return cone
    end
end

WSOSPolyInterpMat(R::Int, U::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpMat(R, U, ipwt, false)

get_nu(cone::WSOSPolyInterpMat) = cone.R * sum(size(ipwtj, 2) for ipwtj in cone.ipwt)

function set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterpMat)
    # sum of diagonal matrices with interpolant polynomial repeating on the diagonal
    idx = 1
    for i in 1:cone.R, j in 1:i
        arr[idx:(idx + cone.U - 1)] .= (i == j) ? 1.0 : 0.0
        idx += cone.U
    end
    return arr
end

_blockrange(inner::Int, outer::Int) = (outer * (inner - 1) + 1):(outer * inner)

# TODO all views can be allocated just once in the cone definition (delete _blockrange too)
function check_in_cone(cone::WSOSPolyInterpMat)
    @timeit "build mat" begin
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

        cone.matfact[j] = cholesky!(Symmetric(mat, :L), Val(true), check = false)
        if !isposdef(cone.matfact[j])
            return false
        end
    end
    end

    @timeit "grad hess" begin
    cone.g .= 0.0
    cone.H .= 0.0
    for j in eachindex(cone.ipwt)
        @timeit "W_inv" W_inv_j = inv(cone.matfact[j])

        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        tmp2 = cone.tmp2
        tmp3 = cone.tmp3

        L = size(ipwtj, 2)
        uo = 0
        for p in 1:cone.R, q in 1:p
            uo += 1
            fact = (p == q) ? 1.0 : rt2
            rinds = _blockrange(p, L)
            cinds = _blockrange(q, L)
            idxs = _blockrange(uo, cone.U)

            for i in 1:cone.U
                cone.g[idxs[i]] -= ipwtj[i, :]' * view(W_inv_j, rinds, cinds) * ipwtj[i, :] * fact
            end

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
                fact = xor(p == q, p2 == q2) ? rt2i : 1.0
                @. cone.H[idxs, idxs2] += tmp2 * tmp3 * fact

                if (p != q) || (p2 != q2)
                    mul!(tmp1j, view(W_inv_j, rinds, cinds2), ipwtj')
                    mul!(tmp2, ipwtj, tmp1j)
                    mul!(tmp1j, view(W_inv_j, cinds, rinds2), ipwtj')
                    mul!(tmp3, ipwtj, tmp1j)
                    @. cone.H[idxs, idxs2] += tmp2 * tmp3 * fact
                end
            end
        end
    end
    end

    ret = @timeit "fact hess" factorize_hess(cone)
    return ret
end
