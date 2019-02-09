#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial matrix cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

mutable struct WSOSPolyInterpMat <: Cone
    use_dual::Bool
    dim::Int
    r::Int
    u::Int
    ipwt::Vector{Matrix{Float64}}
    point::Vector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    F
    mat::Vector{Matrix{Float64}}
    matfact::Vector{CholeskyPivoted{Float64,Array{Float64,2}}}

    function WSOSPolyInterpMat(r::Int, u::Int, ipwt::Vector{Matrix{Float64}}, is_dual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == u
        end
        cone = new()
        cone.use_dual = !is_dual # using dual barrier
        dim = u * div(r * (r + 1), 2)
        cone.dim = dim
        cone.r = r
        cone.u = u
        cone.ipwt = ipwt
        cone.point = similar(ipwt[1], dim)
        cone.g = similar(ipwt[1], dim)
        cone.H = similar(ipwt[1], dim, dim)
        cone.H2 = similar(cone.H)
        cone.mat = [similar(ipwt[1], size(ipwtj, 2) * r, size(ipwtj, 2) * r) for ipwtj in ipwt]
        cone.matfact = Vector{CholeskyPivoted{Float64,Array{Float64,2}}}(undef, length(ipwt))
        return cone
    end
end

function buildmat!(cone::WSOSPolyInterpMat, point::AbstractVector{Float64})
    (R, U) = (cone.r, cone.u)
    for (j, ipwtj) in enumerate(cone.ipwt)
        L = size(ipwtj, 2)
        mat = cone.mat[j]
        mat .= 0.0

        for l in 1:L, k in 1:l
            uo = 0
            for p in 1:R, q in 1:p
                (bp, bq) = ((p - 1) * L, (q - 1) * L)
                val = sum(ipwtj[u, l] * ipwtj[u, k] * point[uo + u] for u in 1:U)
                if p == q
                    mat[bp + l, bq + k] = val
                else
                    mat[bp + l, bq + k] = mat[bp + k, bq + l] = rt2i * val
                end
                uo += U
            end
        end
        cone.matfact[j] = cholesky!(Symmetric(mat, :L), Val(true), check=false)
        if !isposdef(cone.matfact[j])
            return false
        end
    end
    return true
end

function update_gradient_hessian!(cone::WSOSPolyInterpMat, ipwtj::Matrix{Float64}, Winv::Matrix{Float64})
    L = size(ipwtj, 2)
    idx = 0
    for p in 1:cone.r, q in 1:p
        (bp, bq) = ((p - 1) * L, (q - 1) * L)
        for u in 1:cone.u
            idx += 1
            for k in 1:L, l in 1:k
                if k > l
                    Wcomp = Winv[bp + k, bq + l] + Winv[bp + l, bq + k]
                else
                    Wcomp = Winv[bp + k, bq + l]
                end
                if p == q
                    fact = 1.0
                else
                    fact = rt2
                end
                cone.g[idx] -= ipwtj[u, k] * ipwtj[u, l] * Wcomp * fact
            end
            # hessian
            idx2 = 0
            for p2 in 1:cone.r, q2 in 1:p2
                (bp2, bq2) = ((p2 - 1) * L, (q2 - 1) * L)
                for u2 in 1:cone.u
                    idx2 += 1
                    idx2 < idx && continue
                    sum1 = 0.0
                    sum2 = 0.0
                    sum3 = 0.0
                    sum4 = 0.0
                    for k2 in 1:L, l2 in 1:L
                        sum1 += Winv[bp + k2, bp2 + l2] * ipwtj[u, k2] * ipwtj[u2, l2]
                        sum2 += Winv[bq + k2, bq2 + l2] * ipwtj[u, k2] * ipwtj[u2, l2]
                        sum3 += Winv[bp + k2, bq2 + l2] * ipwtj[u, k2] * ipwtj[u2, l2]
                        sum4 += Winv[bq + k2, bp2 + l2] * ipwtj[u, k2] * ipwtj[u2, l2]
                    end
                    sum12 = sum1 * sum2
                    if (p == q) && (p2 == q2)
                        cone.H[idx, idx2] += sum12
                    else
                        sum34 = sum3 * sum4
                        if (p != q) && (p2 != q2)
                            cone.H[idx, idx2] += sum12 + sum34
                        else
                            cone.H[idx, idx2] += rt2i * (sum12 + sum34)
                        end
                    end
                end # u2
            end # p2, q2
        end # u
    end # p, q
    return nothing
end

WSOSPolyInterpMat(r::Int, u::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpMat(r, u, ipwt, false)

get_nu(cone::WSOSPolyInterpMat) = cone.r * sum(size(ipwtj, 2) for ipwtj in cone.ipwt)

function set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterpMat)
    # sum of diagonal matrices with interpolant polynomial repeating on the diagonal
    idx = 1
    for i in 1:cone.r, j in 1:i, u in 1:cone.u
        if i == j
            arr[idx] = 1.0
        else
            arr[idx] = 0.0
        end
        idx += 1
    end
    return arr
end

function check_in_cone(cone::WSOSPolyInterpMat)
    if !(buildmat!(cone, cone.point))
        return false
    end
    cone.g .= 0.0
    cone.H .= 0.0
    for (j, ipwtj) in enumerate(cone.ipwt)
        Winv = inv(cone.matfact[j])
        update_gradient_hessian!(cone, ipwtj, Winv)
    end
    return factorize_hess(cone)
end
