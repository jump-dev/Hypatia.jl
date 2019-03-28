#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial second order cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
=#

mutable struct WSOSPolyInterpSOC <: Cone
    use_dual::Bool
    dim::Int
    R::Int
    U::Int
    ipwt::Vector{Matrix{Float64}}
    point::Vector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    gtry::Vector{Float64}
    Htry::Matrix{Float64}
    H2::Matrix{Float64}
    Hi::Matrix{Float64}
    F
    mat::Vector{Matrix{Float64}}
    matfact::Vector{CholeskyPivoted{Float64, Matrix{Float64}}}
    tmp1::Vector{Matrix{Float64}}
    tmp2::Vector{Matrix{Float64}}
    tmp3::Matrix{Float64}
    tmp4::Vector{Matrix{Float64}}
    lambda::Vector{Vector{Matrix{Float64}}}
    li_lambda::Vector{Vector{Matrix{Float64}}} #TODO unused currently, but could store lambdafact.L \ lambda_i
    blocki_ipwt::Vector{Vector{Vector{Matrix{Float64}}}} # TODO usused currently, but could store Winv(i, j) \ ipwtj
    lambdafact::Vector{CholeskyPivoted{Float64, Matrix{Float64}}}
    diffres

    function WSOSPolyInterpSOC(R::Int, U::Int, ipwt::Vector{Matrix{Float64}}, is_dual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == U
        end
        cone = new()
        cone.use_dual = !is_dual # using dual barrier
        dim = U * R
        cone.dim = dim
        cone.R = R
        cone.U = U
        cone.ipwt = ipwt
        cone.point = similar(ipwt[1], dim)
        cone.g = similar(ipwt[1], dim)
        cone.H = similar(ipwt[1], dim, dim)
        cone.gtry = similar(ipwt[1], dim)
        cone.Htry = similar(ipwt[1], dim, dim)
        cone.H2 = similar(cone.H)
        cone.Hi = similar(cone.H)
        cone.mat = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        cone.matfact = Vector{CholeskyPivoted{Float64, Matrix{Float64}}}(undef, length(ipwt))
        cone.tmp1 = [similar(ipwt[1], size(ipwtj, 2), U) for ipwtj in ipwt]
        cone.tmp2 = [similar(ipwt[1], size(ipwtj, 2), U) for ipwtj in ipwt]
        cone.tmp3 = similar(ipwt[1], U, U)
        cone.tmp4 = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        cone.lambda = [Vector{Matrix{Float64}}(undef, R) for ipwtj in ipwt]
        for j in eachindex(ipwt), r in 1:R
            cone.lambda[j][r] = similar(ipwt[1], size(ipwt[j], 2), size(ipwt[j], 2))
        end
        cone.li_lambda = [Vector{Matrix{Float64}}(undef, R - 1) for ipwtj in ipwt]
        for j in eachindex(ipwt), r in 1:(R - 1)
            cone.li_lambda[j][r] = similar(ipwt[1], size(ipwt[j], 2), size(ipwt[j], 2))
        end
        cone.blocki_ipwt = [Vector{Vector{Matrix{Float64}}}(undef, R) for ipwtj in ipwt]
        for j in eachindex(ipwt), r1 in 1:R
            cone.blocki_ipwt[j][r1] = Vector{Matrix{Float64}}(undef, r1)
            for r2 in 1:r1
                cone.blocki_ipwt[j][r1][r2] = similar(ipwt[1], size(ipwt[j], 2), size(ipwt[j], 2))
            end
        end
        cone.lambdafact = Vector{CholeskyPivoted{Float64, Matrix{Float64}}}(undef, length(ipwt))
        cone.diffres = DiffResults.HessianResult(cone.g)
        return cone
    end
end

WSOSPolyInterpSOC(R::Int, U::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpSOC(R, U, ipwt, false)

get_nu(cone::WSOSPolyInterpSOC) = sum(size(ipwtj, 2) for ipwtj in cone.ipwt)

function set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterpSOC)
    arr .= 0.0
    arr[1:cone.U] .= 1.0
    return arr
end

# ForwardDiff function only TODO remove
function getlambda(point, cone, j)
    ipwtj = cone.ipwt[j]
    L = size(ipwtj, 2)
    mat = similar(point, L, L)

    # first lambda
    point_pq = point[1:cone.U]
    first_lambda = ipwtj' * Diagonal(point_pq) * ipwtj
    mat = Symmetric(first_lambda, :U)

    # minus other lambdas
    uo = cone.U + 1
    for p in 2:cone.R
        point_pq = point[uo:(uo + cone.U - 1)]
        tmp = Symmetric(ipwtj' * Diagonal(point_pq) * ipwtj)
        mat -= Symmetric(tmp * (Symmetric(first_lambda, :U) \ tmp'))
        uo += cone.U
    end
    return Symmetric(mat, :U)
end

# TODO update cone.blocki_ipwt(r1, r2) in place, general form of update is (u1 * inv(matfact) * u2') \ ipwtj
function rank_one_inv_update(cone::WSOSPolyInterpSOC, r1::Int, r2::Int, j::Int)
    lambda_inv = inv(cone.lambdafact[j])
    if (r1 != 1) && (r2 != 1)
        u1 = lambda_inv * Symmetric(cone.lambda[j][r1])
        u2 = lambda_inv * Symmetric(cone.lambda[j][r2])
        return u1 * (cone.matfact[j] \ u2')
    elseif (r1 != 1) && (r2 == 1)
        u1 = lambda_inv * Symmetric(cone.lambda[j][r1])
        return -u1 * Symmetric(inv(cone.matfact[j]))
    elseif (r1 == 1) && (r2 != 1)
        u2 = lambda_inv * Symmetric(cone.lambda[j][r2])
        return -(cone.matfact[j] \ u2')
    else
        return Symmetric(inv(cone.matfact[j]))
    end
end

# TODO should update cone.blocki_ipwt, with Winv(r1, r2) \ ipwtj, return nothing, see above
function mat_inv(cone::WSOSPolyInterpSOC, r1::Int, r2::Int, j::Int)
    ret = rank_one_inv_update(cone, r1, r2, j)
    if (r1 == r2) && (r1 != 1)
        ret += Symmetric(inv(cone.lambdafact[j]))
    end
    return ret
end


function check_in_cone(cone::WSOSPolyInterpSOC)

    # @timeit "build mat" begin
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        lambda = cone.lambda[j]
        li_lambda = cone.li_lambda[j]
        L = size(ipwtj, 2)
        mat = cone.mat[j]

        lambdafact = cone.lambdafact

        # first lambda
        point_pq = cone.point[1:cone.U]
        @. tmp1j = ipwtj' * point_pq'
        mul!(mat, tmp1j, ipwtj)
        lambda[1] .= mat
        # TODO in-place
        lambdafact[j] = cholesky(Symmetric(mat, :L), Val(true), check = false)
        if !isposdef(lambdafact[j])
            return false
        end

        # minus others
        uo = cone.U + 1
        for p in 2:cone.R
            # store lambda
            point_pq = cone.point[uo:(uo + cone.U - 1)] # TODO prealloc
            @. tmp1j = ipwtj' * point_pq'
            mul!(lambda[p], tmp1j, ipwtj)
            # update mat
            li_lambda[p - 1] .= view(lambda[p], lambdafact[j].p, :)
            ldiv!(lambdafact[j].L, li_lambda[p - 1])
            BLAS.syrk!('U', 'T', -1.0, li_lambda[p - 1], 1.0, mat)
            uo += cone.U
        end

        cone.matfact[j] = cholesky!(Symmetric(mat, :U), Val(true), check = false)
        if !isposdef(cone.matfact[j])
            return false
        end
    end
    # end

    fdg = zeros(length(cone.g))
    fdh = zeros(size(cone.H))
    for j in eachindex(cone.ipwt)
        cone.diffres = ForwardDiff.hessian!(cone.diffres, x -> -logdet(getlambda(x, cone, j)), cone.point)
        fdg += DiffResults.gradient(cone.diffres)
        fdh += DiffResults.hessian(cone.diffres)
    end


    # @timeit "grad hess" begin
    cone.g .= 0.0
    cone.H .= 0.0
    for j in eachindex(cone.ipwt)


        Winv(r1, r2) = mat_inv(cone, r1, r2, j)
        # TODO
        # getinv(r1::Int, r2::Int) = cone.blocki_ipwt[j][r1][r2] or cone.blocki_ipwt[j][r2][r1]' if r1

        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        tmp2j = cone.tmp2[j]
        tmp3 = cone.tmp3

        L = size(ipwtj, 2)

        # part of gradient/hessian when p=1
        tmp2j .= view(ipwtj', cone.lambdafact[j].p, :)
        ldiv!(cone.lambdafact[j].L, tmp2j)
        BLAS.syrk!('U', 'T', 1.0, tmp2j, 0.0, tmp3)

        for i in 1:cone.U
            cone.g[i] += tmp3[i, i] * (cone.R - 1)
            for r in 1:cone.R
                cone.g[i] -= ipwtj[i, :]' * Winv(r, r) * ipwtj[i, :]
            end
            for k in 1:i
                cone.H[k, i] -= abs2(tmp3[k, i]) * (cone.R - 1)
            end
        end

        for r in 1:cone.R, s in 1:cone.R
            # TODO
            # mul!(tmp2, ipwtj, getinv(r, s))
            # @.cone.H[idxs, idxs2] += tmp2
            cone.H[1:cone.U, 1:cone.U] += (ipwtj * Winv(r, s) * ipwtj').^2
        end

        for p2 in 2:cone.R
            idxs2 = ((p2 - 1) * cone.U + 1):(p2 * cone.U)
            for r in 1:cone.R
                cone.H[1:cone.U, idxs2] += 2 * (ipwtj * Winv(r, 1) * ipwtj') .* (ipwtj * Winv(r, p2) * ipwtj')
            end
        end

        for p in 2:cone.R
            idxs = ((p - 1) * cone.U + 1):(p * cone.U)
            for i in 1:cone.U
                # TODO view(getinv(p, 1), i, :) etc.
                cone.g[idxs[i]] -= ipwtj[i, :]' * Winv(p, 1) * ipwtj[i, :] + ipwtj[i, :]' * Winv(1, p) * ipwtj[i, :]
            end


            for p2 in 2:cone.R
                idxs2 = ((p2 - 1) * cone.U + 1):(p2 * cone.U)

                # TODO
                # mul!(tmp2, ipwtj, getinv(1, 1))
                # mul!(tmp3, ipwtj, getinv(p, p2))
                # @.cone.H[idxs, idxs2] += tmp2 * tmp3 etc.
                cone.H[idxs, idxs2] += 2 * (ipwtj * Winv(1, 1) * ipwtj') .* (ipwtj * Winv(p, p2) * ipwtj') + 2 * (ipwtj * Winv(p, 1) * ipwtj') .* (ipwtj * Winv(1, p2) * ipwtj')
            end
        end # p
    end # j
    # end
    # @show fdg ./ cone.g
    # @show fdh ./ Symmetric(cone.H, :U)
    # @show fdh - Symmetric(cone.H, :U)
    # @show fdh
    # if !isapprox(fdh * cone.point, -fdg)
    #     @show fdh * cone.point, -fdg
    #     # error()
    # end
    if !isapprox(Symmetric(cone.H, :U) * cone.point, -cone.g)
        @show Symmetric(cone.H, :U) * cone.point, -cone.g
        error()
    end
    # @show Symmetric(cone.H, :U)
    # @show Symmetric(cone.H, :U)

    return factorize_hess(cone)
end
