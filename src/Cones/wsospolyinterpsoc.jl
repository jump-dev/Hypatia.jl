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
    # matfact # CholeskyPivoted{Float64, Matrix{Float64}}
    tmp1::Vector{Matrix{Float64}}
    tmp2::Vector{Matrix{Float64}}
    tmp3::Matrix{Float64}
    tmp4::Vector{Matrix{Float64}}
    # lambda::Vector{Vector{Matrix{Float64}}}
    li_lambda::Vector{Vector{Matrix{Float64}}}
    PlambdaiP::Vector{Vector{Vector{Matrix{Float64}}}}
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
        # cone.matfact = TODO
        cone.tmp1 = [similar(ipwt[1], size(ipwtj, 2), U) for ipwtj in ipwt]
        cone.tmp2 = [similar(ipwt[1], size(ipwtj, 2), U) for ipwtj in ipwt]
        cone.tmp3 = similar(ipwt[1], U, U)
        cone.tmp4 = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        # cone.lambda = [Vector{Matrix{Float64}}(undef, R) for ipwtj in ipwt]
        # for j in eachindex(ipwt), r in 1:R
        #     cone.lambda[j][r] = similar(ipwt[1], size(ipwt[j], 2), size(ipwt[j], 2))
        # end
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

function check_in_cone(cone::WSOSPolyInterpSOC)

    # @timeit "build mat" begin
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        tmp1 = cone.tmp1[j]
        # lambda = cone.lambda[j]
        li_lambda = cone.li_lambda[j]
        PlambdaiP = cone.PlambdaiP[j]
        tmp1 = cone.tmp1[j]
        tmp2 = cone.tmp2[j]
        tmp4 = cone.tmp4[j]
        mat = cone.mat[j]

        lambdafact = cone.lambdafact
        L = size(ipwtj, 2)

        # first lambda
        point_pq = cone.point[1:cone.U]
        @. tmp1 = ipwtj' * point_pq'
        mul!(tmp4, tmp1, ipwtj)
        mat .= tmp4
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

        matfact = cholesky!(Symmetric(mat, :U), Val(true), check = false)
        if !isposdef(matfact)
            return false
        end

        # prep PlambdaiP

        # block-(1,1) is P*inv(mat)*P'
        tmp1 .= view(ipwtj', matfact.p, :)
        ldiv!(matfact.L, tmp1)
        # only on-diagonal PlambdaiP are symmetric, messy to make special cases for these
        PlambdaiP[1][1] = tmp1' * tmp1

        # cache lambda0 \ ipwtj' in tmp1
        tmp1 .= lambdafact[j] \ ipwtj'
        for r in 2:cone.R
            PlambdaiP[r][1] = -ipwtj * li_lambda[r - 1] * (matfact \ ipwtj')
            for r2 in 2:r
                PlambdaiP[r][r2] .= ipwtj * li_lambda[r - 1] * (matfact \ (li_lambda[r2 - 1]' * ipwtj'))
                # mul!(tmp2, li_lambda[r2 - 1]', ipwtj')
                # ldiv!(matfact, tmp2)
                # mul!(tmp5, li_lambda[r - 1], tmp2)
                # mul!(PlambdaiP[r][r2], ipwtj, tmp5)
            end
            PlambdaiP[r][r] .+= ipwtj * tmp1
        end

    end
    # end


    # @timeit "grad hess" begin
    cone.g .= 0.0
    cone.H .= 0.0
    for j in eachindex(cone.ipwt)

        ipwtj = cone.ipwt[j]
        tmp1 = cone.tmp1[j]
        tmp2 = cone.tmp2[j]
        tmp3 = cone.tmp3
        PlambdaiP = cone.PlambdaiP[j]

        L = size(ipwtj, 2)

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

        for r in 1:cone.R
            for s in 1:(r - 1)
                cone.H[1:cone.U, 1:cone.U] .+= PlambdaiP[r][s].^2 + (PlambdaiP[r][s]').^2
            end
            cone.H[1:cone.U, 1:cone.U] .+= PlambdaiP[r][r].^2
        end

        for p2 in 2:cone.R
            idxs2 = ((p2 - 1) * cone.U + 1):(p2 * cone.U)
            for r in 1:p2
                cone.H[1:cone.U, idxs2] += 2 * PlambdaiP[r][1] .* PlambdaiP[p2][r]'
            end
            for r in (p2 + 1):cone.R
                cone.H[1:cone.U, idxs2] += 2 * PlambdaiP[r][1] .* PlambdaiP[r][p2]
            end
        end

        for p in 2:cone.R
            idxs = ((p - 1) * cone.U + 1):(p * cone.U)
            for i in 1:cone.U
                cone.g[idxs[i]] -= 2 * PlambdaiP[p][1][i, i]
            end

            for p2 in p:cone.R
                idxs2 = ((p2 - 1) * cone.U + 1):(p2 * cone.U)
                cone.H[idxs, idxs2] .+= 2 * (PlambdaiP[1][1] .* PlambdaiP[p2][p]' + PlambdaiP[p][1] .* PlambdaiP[p2][1]')
            end
        end # p
    end # j
    # end

    if !isapprox(Symmetric(cone.H, :U) * cone.point, -cone.g)
        @show Symmetric(cone.H, :U) * cone.point, -cone.g
        @show cone.point
        error()
    end

    return factorize_hess(cone)
end
