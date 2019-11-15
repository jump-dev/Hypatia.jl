#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial second order cone parametrized by interpolation points Ps

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

barrier is -logdet(shcur(Lambda)) - logdet(Lambda_11)
if schur(M) = A - B * inv(D) * C
logdet(schur) = logdet(M) - logdet(D) = logdet(Lambda) - (R - 1) * logdet(Lambda_11) since our D is an (R - 1)x(R - 1) block diagonal matrix
=#

mutable struct WSOSPolyInterpSOC{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    R::Int
    U::Int
    Ps::Vector{Matrix{T}}
    point::Vector{T}

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

    gtry::Vector{T} # TODO remove
    Htry::Matrix{T} # TODO remove
    H2::Matrix{T} # TODO remove

    mat::Vector{Matrix{T}}
    matfact::Vector
    tmp1::Vector{Matrix{T}}
    tmp2::Vector{Matrix{T}}
    tmp3::Matrix{T}
    tmp4::Vector{Matrix{T}}

    Λi_Λ::Vector{Vector{Matrix{T}}}
    tmpLU::Vector{Matrix{T}}
    PΛiPs::Vector{Vector{Vector{Matrix{T}}}}
    lambdafact::Vector

    function WSOSPolyInterpSOC{T}(
        R::Int,
        U::Int,
        Ps::Vector{Matrix{T}},
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        for Psj in Ps
            @assert size(Psj, 1) == U
        end
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        dim = U * R
        cone.dim = dim
        cone.R = R
        cone.U = U
        cone.Ps = Ps
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

WSOSPolyInterpSOC{T}(R::Int, U::Int, Ps::Vector{Matrix{T}}) where {T <: Real} = WSOSPolyInterpSOC{T}(R, U, Ps, false)

function setup_data(cone::WSOSPolyInterpSOC{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    U = cone.U
    R = cone.R
    Ps = cone.Ps

    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)

    cone.mat = [similar(cone.grad, size(Psj, 2), size(Psj, 2)) for Psj in Ps]
    cone.matfact = Vector{Any}(undef, length(Ps))
    cone.tmp1 = [similar(cone.grad, size(Psj, 2), U) for Psj in Ps]
    cone.tmp2 = [similar(cone.grad, size(Psj, 2), U) for Psj in Ps]
    cone.tmp3 = similar(cone.grad, U, U)
    cone.tmp4 = [similar(cone.grad, size(Psj, 2), size(Psj, 2)) for Psj in Ps]
    cone.Λi_Λ = [Vector{Matrix{T}}(undef, R - 1) for Psj in Ps]
    for j in eachindex(Ps), r in 1:(R - 1)
        cone.Λi_Λ[j][r] = similar(cone.grad, size(Ps[j], 2), size(Ps[j], 2))
    end
    cone.tmpLU = [similar(cone.grad, size(Psj, 2), U) for Psj in Ps]
    cone.PΛiPs = [Vector{Vector{Matrix{T}}}(undef, R) for Psj in Ps]
    for j in eachindex(Ps), r1 in 1:R
        cone.PΛiPs[j][r1] = Vector{Matrix{T}}(undef, r1)
        for r2 in 1:r1
            cone.PΛiPs[j][r1][r2] = similar(cone.grad, U, U)
        end
    end
    cone.lambdafact = Vector{Any}(undef, length(Ps))
    return
end

get_nu(cone::WSOSPolyInterpSOC) = 2 * sum(size(Psj, 2) for Psj in cone.Ps)

function set_initial_point(arr::AbstractVector{T}, cone::WSOSPolyInterpSOC{T}) where {T <: Real}
    arr .= zero(T)
    arr[1:cone.U] .= one(T)
    return arr
end

# TODO cleanup experimental code
function update_feas(cone::WSOSPolyInterpSOC)
    @assert !cone.feas_updated
    cone.is_feas = true
    for j in eachindex(cone.Ps)
        Psj = cone.Ps[j]
        Λi_Λ = cone.Λi_Λ[j]
        tmp1 = cone.tmp1[j]
        tmp2 = cone.tmp2[j]
        tmp4 = cone.tmp4[j]
        mat = cone.mat[j]
        lambdafact = cone.lambdafact
        matfact = cone.matfact
        L = size(Psj, 2)

        # first lambda
        point_pq = cone.point[1:cone.U]
        @. tmp1 = Psj' * point_pq'
        mul!(tmp4, tmp1, Psj)
        mat .= tmp4
        lambdafact[j] = cholesky!(Symmetric(tmp4, :L), check = false)
        if !isposdef(lambdafact[j])
            cone.is_feas = false
            break
        end

        # minus others
        uo = cone.U + 1
        for r in 2:cone.R
            point_pq = cone.point[uo:(uo + cone.U - 1)] # TODO prealloc
            @. tmp1 = Psj' * point_pq'
            tmp4 = tmp1 * Psj
            # mul!(tmp4, tmp1, Psj)

            # avoiding lambdafact.L \ lambda because lambdafact \ lambda is useful later
            Λi_Λ[r - 1] .= lambdafact[j] \ tmp4
            mat -= tmp4 * Λi_Λ[r - 1]
            uo += cone.U
        end

        matfact[j] = cholesky!(Symmetric(mat, :U), check = false)
        if !isposdef(matfact[j])
            cone.is_feas = false
            break
        end
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::WSOSPolyInterpSOC{T}) where {T}
    @assert cone.is_feas
    cone.grad .= 0
    for j in eachindex(cone.Ps)
        Psj = cone.Ps[j]
        tmp1 = cone.tmp1[j]
        tmp2 = cone.tmp2[j]
        PΛiPs = cone.PΛiPs[j]
        LUj = cone.tmpLU[j]
        Λi_Λ = cone.Λi_Λ[j]
        lambdafact = cone.lambdafact
        matfact = cone.matfact

        # prep PΛiPs
        # block-(1,1) is P * inv(mat) * P'
        copyto!(tmp1, Psj')
        ldiv!(LowerTriangular(matfact[j].L), tmp1)
        mul!(PΛiPs[1][1], tmp1', tmp1)

        # cache lambda0 \ Psj' in tmp1
        tmp1 .= lambdafact[j] \ Psj'
        # get all the PΛiPs that are in row one or on the diagonal
        for r in 2:cone.R
            PΛiPs[r][1] = -Psj * Λi_Λ[r - 1] * (matfact[j] \ Psj')
            # PΛiPs[r][r] .= Symmetric(Psj * Λi_Λ[r - 1] * (matfact[j] \ (Λi_Λ[r - 1]' * Psj')), :U)
            mul!(tmp2, Λi_Λ[r - 1]', Psj')
            ldiv!(matfact[j].L, tmp2)
            mul!(PΛiPs[r][r], tmp2', tmp2)
            PΛiPs[r][r] .+= Symmetric(Psj * tmp1, :U)
        end

        # get half of P * inv(Λ_11) * P for (1, 1) hessian block and first gradient block
        copyto!(LUj, Psj')
        ldiv!(LowerTriangular(cone.lambdafact[j].L), LUj)

        # (1, 1)-block
        # gradient is diag of sum(-PΛiPs[i][i] for i in 1:R) + (R - 1) * Lambda_11 - Lambda_11
        for i in 1:cone.U
            cone.grad[i] += sum(abs2, view(LUj, :, i)) * (cone.R - 2)
            for r in 1:cone.R
                cone.grad[i] -= PΛiPs[r][r][i, i]
            end
        end
        idx = cone.U + 1
        for r in 2:cone.R, i in 1:cone.U
            cone.grad[idx] -= 2 * PΛiPs[r][1][i, i]
            idx += 1
        end
    end # j

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSPolyInterpSOC)
    @assert cone.grad_updated
    cone.hess .= 0

    for j in eachindex(cone.Ps)
        Psj = cone.Ps[j]
        tmp3 = cone.tmp3
        PΛiPs = cone.PΛiPs[j]
        LUj = cone.tmpLU[j]
        Λi_Λ = cone.Λi_Λ[j]
        matfact = cone.matfact

        # get the PΛiPs not calculated in update_grad
        for r in 2:cone.R, r2 in 2:(r - 1)
            PΛiPs[r][r2] .= Psj * Λi_Λ[r - 1] * (matfact[j] \ (Λi_Λ[r2 - 1]' * Psj'))
        end

        # tmp3 = P * inv(Lambda_11) * P'
        mul!(tmp3, LUj', LUj)
        @inbounds for i in 1:cone.U, k in 1:i
            cone.hess.data[k, i] -= abs2(tmp3[k, i]) * (cone.R - 2)
        end

        cone.hess.data[1:cone.U, 1:cone.U] .+= Symmetric(PΛiPs[1][1], :U).^2
        for r in 2:cone.R
            idxs2 = ((r - 1) * cone.U + 1):(r * cone.U)
            for s in 1:(r - 1)
                # block (1,1)
                tmp3 .= PΛiPs[r][s].^2
                cone.hess.data[1:cone.U, 1:cone.U] .+= Symmetric(tmp3 + tmp3', :U)
                # blocks (1,r)
                cone.hess.data[1:cone.U, idxs2] += 2 * PΛiPs[s][1] .* PΛiPs[r][s]'
            end
            # block (1,1)
            cone.hess.data[1:cone.U, 1:cone.U] .+= PΛiPs[r][r].^2
            # blocks (1,r)
            cone.hess.data[1:cone.U, idxs2] += 2 * PΛiPs[r][1] .* Symmetric(PΛiPs[r][r], :U)
            # blocks (1,r)
            for s in (r + 1):cone.R
                cone.hess.data[1:cone.U, idxs2] += 2 * PΛiPs[s][1] .* PΛiPs[s][r]
            end

            # blocks (r, r2)
            idxs = ((r - 1) * cone.U + 1):(r * cone.U)
            cone.hess.data[idxs, idxs2] .+= 2 * Symmetric(Symmetric(PΛiPs[1][1], :U) .* Symmetric(PΛiPs[r][r], :U) + PΛiPs[r][1] .* PΛiPs[r][1]', :U)
            for r2 in (r + 1):cone.R
                idxs2 = ((r2 - 1) * cone.U + 1):(r2 * cone.U)
                cone.hess.data[idxs, idxs2] .+= 2 * (PΛiPs[1][1] .* PΛiPs[r2][r]' + PΛiPs[r][1] .* PΛiPs[r2][1]')
            end
        end

    end
    cone.hess_updated = true
    return cone.hess
end



;
