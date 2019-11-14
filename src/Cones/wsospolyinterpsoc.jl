#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial second order cone parametrized by interpolation points Ps

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
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
    li_lambda::Vector{Vector{Matrix{T}}}
    PlambdaiP::Vector{Vector{Vector{Matrix{T}}}}
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
    cone.li_lambda = [Vector{Matrix{T}}(undef, R - 1) for Psj in Ps]
    for j in eachindex(Ps), r in 1:(R - 1)
        cone.li_lambda[j][r] = similar(cone.grad, size(Ps[j], 2), size(Ps[j], 2))
    end
    cone.PlambdaiP = [Vector{Vector{Matrix{T}}}(undef, R) for Psj in Ps]
    for j in eachindex(Ps), r1 in 1:R
        cone.PlambdaiP[j][r1] = Vector{Matrix{T}}(undef, r1)
        for r2 in 1:r1
            cone.PlambdaiP[j][r1][r2] = similar(cone.grad, U, U)
        end
    end
    cone.lambdafact = Vector{Any}(undef, length(Ps))
    return
end

get_nu(cone::WSOSPolyInterpSOC) = sum(size(Psj, 2) for Psj in cone.Ps)

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
        li_lambda = cone.li_lambda[j]
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
        lambdafact[j] = cholesky!(Symmetric(tmp4, :L), check = false, Val(true)) # TODO don't pivot?
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
            li_lambda[r - 1] .= lambdafact[j] \ tmp4
            mat -= tmp4 * li_lambda[r - 1]
            uo += cone.U
        end

        matfact[j] = cholesky!(Symmetric(mat, :U), check = false, Val(true)) # TODO don't pivot?
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
    cone.hess .= 0
    for j in eachindex(cone.Ps)
        Psj = cone.Ps[j]
        tmp1 = cone.tmp1[j]
        tmp2 = cone.tmp2[j]
        tmp3 = cone.tmp3
        PlambdaiP = cone.PlambdaiP[j]
        li_lambda = cone.li_lambda[j]
        lambdafact = cone.lambdafact
        matfact = cone.matfact

        # prep PlambdaiP
        # block-(1,1) is P*inv(mat)*P'
        tmp1 .= view(Psj', matfact[j].p, :)
        ldiv!(matfact[j].L, tmp1)
        PlambdaiP[1][1] = Symmetric(tmp1' * tmp1)

        # cache lambda0 \ Psj' in tmp1
        tmp1 .= lambdafact[j] \ Psj'
        for r in 2:cone.R
            PlambdaiP[r][1] = -Psj * li_lambda[r - 1] * (matfact[j] \ Psj')
            for r2 in 2:(r - 1)
                PlambdaiP[r][r2] .= Psj * li_lambda[r - 1] * (matfact[j] \ (li_lambda[r2 - 1]' * Psj'))
                # mul!(tmp2, li_lambda[r2 - 1]', Psj')
                # ldiv!(matfact[j], tmp2)
                # mul!(tmp5, li_lambda[r - 1], tmp2)
                # mul!(PlambdaiP[r][r2], Psj, tmp5)
            end
            # PlambdaiP[r][r] .= Symmetric(Psj * li_lambda[r - 1] * (matfact[j] \ (li_lambda[r - 1]' * Psj')), :U)
            mul!(tmp2, li_lambda[r - 1]', Psj')
            tmp2 .= view(tmp2, matfact[j].p, :)
            ldiv!(matfact[j].L, tmp2)
            BLAS.syrk!('U', 'T', one(T), tmp2, zero(T), PlambdaiP[r][r])
            PlambdaiP[r][r] .+= Symmetric(Psj * tmp1, :U)
        end

        # part of gradient/hessian when p=1
        tmp2 .= view(Psj', cone.lambdafact[j].p, :)
        ldiv!(cone.lambdafact[j].L, tmp2)
        BLAS.syrk!('U', 'T', one(T), tmp2, zero(T), tmp3)

        for i in 1:cone.U
            cone.grad[i] += tmp3[i, i] * (cone.R - 1)
            for r in 1:cone.R
                cone.grad[i] -= PlambdaiP[r][r][i, i]
            end
            for k in 1:i
                cone.hess.data[k, i] -= abs2(tmp3[k, i]) * (cone.R - 1)
            end
        end

        cone.hess.data[1:cone.U, 1:cone.U] .+= Symmetric(PlambdaiP[1][1], :U).^2
        for r in 2:cone.R
            idxs2 = ((r - 1) * cone.U + 1):(r * cone.U)
            for s in 1:(r - 1)
                # block (1,1)
                tmp3 .= PlambdaiP[r][s].^2
                cone.hess.data[1:cone.U, 1:cone.U] .+= Symmetric(tmp3 + tmp3', :U)
                # blocks (1,r)
                cone.hess.data[1:cone.U, idxs2] += 2 * PlambdaiP[s][1] .* PlambdaiP[r][s]'
            end
            # block (1,1)
            cone.hess.data[1:cone.U, 1:cone.U] .+= PlambdaiP[r][r].^2
            # blocks (1,r)
            cone.hess.data[1:cone.U, idxs2] += 2 * PlambdaiP[r][1] .* Symmetric(PlambdaiP[r][r], :U)
            # blocks (1,r)
            for s in (r + 1):cone.R
                cone.hess.data[1:cone.U, idxs2] += 2 * PlambdaiP[s][1] .* PlambdaiP[s][r]
            end

            # blocks (r, r2)
            idxs = ((r - 1) * cone.U + 1):(r * cone.U)
            for i in 1:cone.U
                cone.grad[idxs[i]] -= 2 * PlambdaiP[r][1][i, i]
            end
            cone.hess.data[idxs, idxs2] .+= 2 * Symmetric(Symmetric(PlambdaiP[1][1], :U) .* Symmetric(PlambdaiP[r][r], :U) + PlambdaiP[r][1] .* PlambdaiP[r][1]', :U)
            for r2 in (r + 1):cone.R
                idxs2 = ((r2 - 1) * cone.U + 1):(r2 * cone.U)
                cone.hess.data[idxs, idxs2] .+= 2 * (PlambdaiP[1][1] .* PlambdaiP[r2][r]' + PlambdaiP[r][1] .* PlambdaiP[r2][1]')
            end
        end
    end # j

    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::WSOSPolyInterpSOC)
    @assert cone.grad_updated
    cone.hess_updated = true
    return cone.hess
end



;
