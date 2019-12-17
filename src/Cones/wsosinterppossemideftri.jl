#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial positive semidefinite cone parametrized by interpolation matrices Ps
certifies that a polynomial valued R x R matrix is in the positive semidefinite cone for all x in the domain defined by Ps

dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
and "Semidefinite Characterization of Sum-of-Squares Cones in Algebras" by D. Papp and F. Alizadeh
=#

mutable struct WSOSInterpPosSemidefTri{T <: Real} <: Cone{T}
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
    hess_fact_cache

    rt2::T
    rt2i::T
    slice::Vector{T}
    LL::Vector{Symmetric{T, Matrix{T}}}
    ΛFs::Vector
    ΛFP::Vector{Matrix{T}}
    LUs::Vector{Matrix{T}}
    UU1::Matrix{T}
    UU2::Matrix{T}
    PlambdaP::Vector{Matrix{T}}

    function WSOSInterpPosSemidefTri{T}(
        R::Int,
        U::Int,
        Ps::Vector{Matrix{T}},
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        for Pj in Ps
            @assert size(Pj, 1) == U
        end
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = U * div(R * (R + 1), 2)
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

WSOSInterpPosSemidefTri{T}(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSInterpPosSemidefTri{T}(R, U, Ps, false)

function setup_data(cone::WSOSInterpPosSemidefTri{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps
    cone.point = zeros(T, dim)
    cone.grad = similar(cone.point)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.rt2 = sqrt(T(2))
    cone.rt2i = inv(cone.rt2)
    cone.slice = Vector{T}(undef, U)
    cone.LL = [Symmetric(zeros(T, size(Pj, 2) * R, size(Pj, 2) * R), :L) for Pj in Ps]
    cone.ΛFs = Vector{Any}(undef, length(Ps))
    cone.ΛFP = [Matrix{T}(undef, R * size(Pj, 2), R * U) for Pj in Ps]
    cone.LUs = [Matrix{T}(undef, size(Pj, 2), U) for Pj in Ps]
    cone.UU1 = Matrix{T}(undef, U, U)
    cone.UU2 = similar(cone.UU1)
    cone.PlambdaP = [zeros(T, R * U, R * U) for _ in eachindex(Ps)]
    return
end

get_nu(cone::WSOSInterpPosSemidefTri) = cone.R * sum(size(Pj, 2) for Pj in cone.Ps)

function set_initial_point(arr::AbstractVector, cone::WSOSInterpPosSemidefTri)
    arr .= 0
    idx = 1
    @inbounds for i in 1:cone.R
        arr[idx:(idx + cone.U - 1)] .= 1
        idx += (i + 1) * cone.U
    end
    return arr
end

function update_feas(cone::WSOSInterpPosSemidefTri)
    @assert !cone.feas_updated

    cone.is_feas = true
    @inbounds for j in eachindex(cone.Ps)
        Pj = cone.Ps[j]
        LU = cone.LUs[j]
        L = size(Pj, 2)
        Λ = cone.LL[j]
        uo = rowo = 1
        @inbounds for p in 1:cone.R
            colo = 1
            @inbounds for q in 1:p
                fact = (p == q ? 1 : cone.rt2i)
                cone.slice .= view(cone.point, uo:(uo + cone.U - 1)) * fact
                mul!(LU, Pj', Diagonal(cone.slice))
                mul!(view(Λ.data, rowo:(rowo + L - 1), colo:(colo + L - 1)), LU, Pj)
                uo += cone.U
                colo += L
            end
            rowo += L
        end
        cone.ΛFs[j] = cholesky!(Λ, check = false)
        if !isposdef(cone.ΛFs[j])
            cone.is_feas = false
            break
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSInterpPosSemidefTri)
    @assert is_feas(cone)
    R = cone.R
    U = cone.U

    cone.grad .= 0
    @inbounds for j in eachindex(cone.Ps)
        Pj = cone.Ps[j]
        L = size(Pj, 2)
        get_PlambdaP(cone, j)
        PlambdaP = Symmetric(cone.PlambdaP[j], :U)
        idx = rowo = 1
        @inbounds for p in 1:R, q in 1:p
            fact = (p == q ? 1 : cone.rt2)
            @inbounds for i in 1:cone.U
                cone.grad[idx] -= PlambdaP[(p - 1) * U + i, (q - 1) * U + i] * fact
                idx += 1
            end
        end
    end

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSInterpPosSemidefTri)
    @assert is_feas(cone)

    cone.hess .= 0
    U = cone.U
    @inbounds for j in eachindex(cone.Ps)
        PlambdaP = Symmetric(cone.PlambdaP[j], :U)
        uo = 0
        @inbounds for p in 1:cone.R, q in 1:p
            uo += 1
            fact = (p == q) ? 1 : cone.rt2
            rinds = (U * (p - 1) + 1):(U * p)
            cinds = (U * (q - 1) + 1):(U * q)
            idxs = (U * (uo - 1) + 1):(U * uo)

            uo2 = 0
            @inbounds for p2 in 1:cone.R, q2 in 1:p2
                uo2 += 1
                if uo2 < uo
                    continue
                end
                rinds2 = (U * (p2 - 1) + 1):(U * p2)
                cinds2 = (U * (q2 - 1) + 1):(U * q2)
                idxs2 = (U * (uo2 - 1) + 1):(U * uo2)

                fact = xor(p == q, p2 == q2) ? cone.rt2i : 1
                @. cone.hess.data[idxs, idxs2] += PlambdaP[rinds, rinds2] * PlambdaP[cinds, cinds2] * fact

                @inbounds if (p != q) || (p2 != q2)
                    @. cone.hess.data[idxs, idxs2] += PlambdaP[rinds, cinds2] * PlambdaP[cinds, rinds2] * fact
                end
            end # p2, q2
        end #p, q
    end # Ps

    cone.hess_updated = true
    return cone.hess
end

function get_PlambdaP(cone::WSOSInterpPosSemidefTri{T}, j::Int) where {T}
    R = cone.R
    U = cone.U
    L = size(cone.Ps[j], 2)
    PlambdaP = cone.PlambdaP[j]
    ΛF = cone.ΛFs[j].L
    LU = cone.LUs[j]
    ΛFP = cone.ΛFP[j]

    # given cholesky factorization L factor ΛF, get ΛFP = ΛF * kron(I, P')
    @inbounds for k in 1:R
        # kth column block of ΛFP
        cols = ((k - 1) * U + 1):(k * U)
        # block (k, k) of L factor of ΛF
        ΛF_kk = view(ΛF, ((k - 1) * L + 1):(k * L), ((k - 1) * L + 1):(k * L))
        # block (k, k) of ΛFP
        ΛFP_kk = view(ΛFP, ((k - 1) * L + 1):(k * L), cols)
        # ΛFP_kk = ΛF_kk \ P'
        copyto!(ΛFP_kk, cone.Ps[j]')
        ldiv!(LowerTriangular(ΛF_kk), ΛFP_kk)
        # to get off-diagonals in ΛFP, subtract known blocks aggregated in LU
        @inbounds for r in (k + 1):R
            LU .= 0
            @inbounds for s in k:(r - 1)
                # block (s, k) of ΛFP
                ΛFP_ks = view(ΛFP, ((s - 1) * L + 1):(s * L), cols)
                # block (r, s) of L factor of ΛF
                ΛF_rs = view(ΛF, ((r - 1) * L + 1):(r * L), ((s - 1) * L + 1):(s * L))
                mul!(LU, ΛF_rs, ΛFP_ks, -one(T), one(T))
            end
            # block (r, r) of ΛF
            ΛF_rr = view(ΛF, ((r - 1) * L + 1):(r * L), ((r - 1) * L + 1):(r * L))
            # block (r, k) of ΛFP
            ΛFP_rk = view(ΛFP, ((r - 1) * L + 1):(r * L), cols)
            copyto!(ΛFP_rk, LU)
            ldiv!(LowerTriangular(ΛF_rr), ΛFP_rk)
        end
    end

    # PlambdaP = ΛFP' * ΛFP
    @inbounds for r in 1:R
        rinds = ((r - 1) * U + 1):(r * U)
        @inbounds for s in r:R
            cinds = ((s - 1) * U + 1):(s * U)
            # since ΛFP is block lower triangular rows only from max(i,j) start making a nonzero contribution to the product
            mulrange = ((s - 1) * L + 1):(L * R)
            @views mul!(PlambdaP[rinds, cinds], ΛFP[mulrange, rinds]',  ΛFP[mulrange, cinds])
        end
    end

    return PlambdaP
end
