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
    tmp1::Vector{Matrix{Float64}}
    tmp2::Matrix{Float64}
    tmp3::Matrix{Float64}

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
        cone.tmp1 = [similar(ipwt[1], size(ipwtj, 2), u) for ipwtj in ipwt]
        cone.tmp2 = similar(ipwt[1], u, u)
        cone.tmp3 = similar(ipwt[1], u, u)
        return cone
    end
end

_blockrange(inner::Int, outer::Int) = (outer * (inner - 1) + 1):(outer * inner)

function buildmat!(cone::WSOSPolyInterpMat, point::AbstractVector{Float64})
    (R, U) = (cone.r, cone.u)
    tmp2 = cone.tmp2
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        L = size(ipwtj, 2)
        mat = cone.mat[j]
        mat .= 0.0

        uo = 0
        for p in 1:R, q in 1:p # seems like blocks unrelated could be v parallel
            (p == q) ? fact = 1.0 : fact = rt2i
            rinds = _blockrange(p, L)
            cinds = _blockrange(q, L)
            @. tmp1j = ipwtj' * cone.point[uo + 1:uo + U]' * fact
            mul!(view(mat, rinds, cinds), tmp1j, ipwtj)
            uo += U
        end
        cone.matfact[j] = cholesky!(Symmetric(mat, :L), Val(true), check=false)
        if !isposdef(cone.matfact[j])
            return false
        end
    end
    return true
end

function update_gradient_hessian!(cone::WSOSPolyInterpMat, j::Int, Winv::Matrix{Float64})
    ipwtj = cone.ipwt[j]
    tmp1j = cone.tmp1[j]
    tmp2 = cone.tmp2
    tmp3 = cone.tmp3

    L = size(ipwtj, 2)

    uo = 0
    for p in 1:cone.r, q in 1:p
        uo += 1
        (p == q) ? fact = 1.0 : fact = rt2
        rinds = _blockrange(p, L)
        cinds = _blockrange(q, L)
        idxs = _blockrange(uo, cone.u)
        mul!(tmp1j, view(Winv, rinds, cinds), ipwtj')
        mul!(tmp2, ipwtj, tmp1j)
        cone.g[idxs] .-= diag(tmp2) * fact
        uo2 = 0
        for p2 in 1:cone.r, q2 in 1:p2
            uo2 += 1
            uo2 < uo && continue
            rinds2 = _blockrange(p2, L)
            cinds2 = _blockrange(q2, L)
            idxs2 = _blockrange(uo2, cone.u)

            mul!(tmp1j, view(Winv, rinds, rinds2), ipwtj')
            mul!(tmp2, ipwtj, tmp1j)
            mul!(tmp1j, view(Winv, cinds, cinds2), ipwtj')
            mul!(tmp3, ipwtj, tmp1j)
            xor(p == q, p2 == q2) ? fact = rt2i : fact = 1.0
            @. cone.H[idxs, idxs2] += tmp2 * tmp3 * fact

            if (p != q) || (p2 != q2)
                mul!(tmp1j, view(Winv, rinds, cinds2), ipwtj')
                mul!(tmp2, ipwtj, tmp1j)
                mul!(tmp1j, view(Winv, cinds, rinds2), ipwtj')
                mul!(tmp3, ipwtj, tmp1j)
                @. cone.H[idxs, idxs2] += tmp2 * tmp3 * fact
            end
        end
    end
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
        update_gradient_hessian!(cone, j, Winv)
    end
    return factorize_hess(cone)
end
