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
    li_lambda::Vector{Vector{Matrix{Float64}}}
    PlambdaiP::Vector{Vector{Vector{Matrix{Float64}}}}
    lambdafact::Vector{CholeskyPivoted{Float64, Matrix{Float64}}}

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
        cone.li_lambda = [Vector{Matrix{Float64}}(undef, R - 1) for ipwtj in ipwt]
        for j in eachindex(ipwt), r in 1:(R - 1)
            cone.li_lambda[j][r] = similar(ipwt[1], size(ipwt[j], 2), size(ipwt[j], 2))
        end
        cone.PlambdaiP = [Vector{Vector{Matrix{Float64}}}(undef, R) for ipwtj in ipwt]
        for j in eachindex(ipwt), r1 in 1:R
            cone.PlambdaiP[j][r1] = Vector{Matrix{Float64}}(undef, r1)
            for r2 in 1:r1
                cone.PlambdaiP[j][r1][r2] = similar(ipwt[1], U, U)
            end
        end
        cone.lambdafact = Vector{CholeskyPivoted{Float64, Matrix{Float64}}}(undef, length(ipwt))
        return cone
    end
end

WSOSPolyInterpSOC(R::Int, U::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpSOC(R, U, ipwt, false)

get_nu(cone::WSOSPolyInterpSOC) = sum(size(ipwtj, 2) for ipwtj in cone.ipwt) #* cone.R

function set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterpSOC)
    arr .= 0.0
    arr[1:cone.U] .= 1.0
    return arr
end

function check_in_cone(cone::WSOSPolyInterpSOC)
    # @timeit to "build mat" begin
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        li_lambda = cone.li_lambda[j]
        tmp1 = cone.tmp1[j]
        tmp2 = cone.tmp2[j]
        tmp4 = cone.tmp4[j]
        mat = cone.mat[j]
        lambdafact = cone.lambdafact
        matfact = cone.matfact
        L = size(ipwtj, 2)

        # first lambda
        point_pq = cone.point[1:cone.U]
        @. tmp1 = ipwtj' * point_pq'
        mul!(tmp4, tmp1, ipwtj)
        mat .= tmp4
        # @show eigen(Symmetric(tmp4, :L)).values
        lambdafact[j] = cholesky!(Symmetric(tmp4, :L), Val(true), check = false)


        if !isposdef(lambdafact[j])
            return false
        end

        # minus others
        uo = cone.U + 1
        for r in 2:cone.R
            point_pq = cone.point[uo:(uo + cone.U - 1)] # TODO prealloc
            @. tmp1 = ipwtj' * point_pq'
            tmp4 = tmp1 * ipwtj
            # mul!(tmp4, tmp1, ipwtj)

            # avoiding lambdafact.L \ lambda because lambdafact \ lambda is useful later
            li_lambda[r - 1] .= lambdafact[j] \ tmp4
            mat -= tmp4 * li_lambda[r - 1]
            uo += cone.U
        end

        # @show eigen(Symmetric(mat, :U)).values

        matfact[j] = cholesky!(Symmetric(mat, :U), Val(true), check = false)
        if !isposdef(matfact[j])
            return false
        end
    end
    # end

    # @timeit to "grad hess" begin
    cone.g .= 0.0
    cone.H .= 0.0
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        tmp1 = cone.tmp1[j]
        tmp2 = cone.tmp2[j]
        tmp3 = cone.tmp3
        PlambdaiP = cone.PlambdaiP[j]
        li_lambda = cone.li_lambda[j]
        lambdafact = cone.lambdafact
        matfact = cone.matfact

        # @timeit to "build plp" begin

        # prep PlambdaiP
        # block-(1,1) is P*inv(mat)*P'
        tmp1 .= view(ipwtj', matfact[j].p, :)
        ldiv!(matfact[j].L, tmp1)
        PlambdaiP[1][1] = Symmetric(tmp1' * tmp1)

        # cache lambda0 \ ipwtj' in tmp1
        tmp1 .= lambdafact[j] \ ipwtj'
        for r in 2:cone.R
            PlambdaiP[r][1] = -ipwtj * li_lambda[r - 1] * (matfact[j] \ ipwtj')
            for r2 in 2:(r - 1)
                PlambdaiP[r][r2] .= ipwtj * li_lambda[r - 1] * (matfact[j] \ (li_lambda[r2 - 1]' * ipwtj'))
                # mul!(tmp2, li_lambda[r2 - 1]', ipwtj')
                # ldiv!(matfact[j], tmp2)
                # mul!(tmp5, li_lambda[r - 1], tmp2)
                # mul!(PlambdaiP[r][r2], ipwtj, tmp5)
            end
            # PlambdaiP[r][r] .= Symmetric(ipwtj * li_lambda[r - 1] * (matfact[j] \ (li_lambda[r - 1]' * ipwtj')), :U)
            mul!(tmp2, li_lambda[r - 1]', ipwtj')
            tmp2 .= view(tmp2, matfact[j].p, :)
            ldiv!(matfact[j].L, tmp2)
            BLAS.syrk!('U', 'T', 1.0, tmp2, 0.0, PlambdaiP[r][r])
            PlambdaiP[r][r] .+= Symmetric(ipwtj * tmp1, :U)
        end

        # end

        # part of gradient/hessian when p=1
        tmp2 .= view(ipwtj', cone.lambdafact[j].p, :)
        ldiv!(cone.lambdafact[j].L, tmp2)
        BLAS.syrk!('U', 'T', 1.0, tmp2, 0.0, tmp3)

        for i in 1:cone.U
            cone.g[i] += tmp3[i, i] * (cone.R - 1)
            for r in 1:cone.R
                cone.g[i] -= PlambdaiP[r][r][i, i]
            end
            for k in 1:i
                cone.H[k, i] -= abs2(tmp3[k, i]) * (cone.R - 1)
            end
        end

        cone.H[1:cone.U, 1:cone.U] .+= Symmetric(PlambdaiP[1][1], :U).^2
        for r in 2:cone.R
            idxs2 = ((r - 1) * cone.U + 1):(r * cone.U)
            for s in 1:(r - 1)
                # block (1,1)
                tmp3 .= PlambdaiP[r][s].^2
                cone.H[1:cone.U, 1:cone.U] .+= Symmetric(tmp3 + tmp3', :U)
                # blocks (1,r)
                cone.H[1:cone.U, idxs2] += 2 * PlambdaiP[s][1] .* PlambdaiP[r][s]'
            end
            # block (1,1)
            cone.H[1:cone.U, 1:cone.U] .+= PlambdaiP[r][r].^2
            # blocks (1,r)
            cone.H[1:cone.U, idxs2] += 2 * PlambdaiP[r][1] .* Symmetric(PlambdaiP[r][r], :U)
            # blocks (1,r)
            for s in (r + 1):cone.R
                cone.H[1:cone.U, idxs2] += 2 * PlambdaiP[s][1] .* PlambdaiP[s][r]
            end

            # blocks (r, r2)
            idxs = ((r - 1) * cone.U + 1):(r * cone.U)
            for i in 1:cone.U
                cone.g[idxs[i]] -= 2 * PlambdaiP[r][1][i, i]
            end
            cone.H[idxs, idxs2] .+= 2 * Symmetric(Symmetric(PlambdaiP[1][1], :U) .* Symmetric(PlambdaiP[r][r], :U) + PlambdaiP[r][1] .* PlambdaiP[r][1]', :U)
            for r2 in (r + 1):cone.R
                idxs2 = ((r2 - 1) * cone.U + 1):(r2 * cone.U)
                cone.H[idxs, idxs2] .+= 2 * (PlambdaiP[1][1] .* PlambdaiP[r2][r]' + PlambdaiP[r][1] .* PlambdaiP[r2][1]')
            end
        end
    end # j
    # end

    # if !isapprox(Symmetric(cone.H, :U) * cone.point, -cone.g)
    #     @show Symmetric(cone.H, :U) * cone.point, -cone.g
    #     @show cone.point
    #     error()
    # end

    return factorize_hess(cone)
end
