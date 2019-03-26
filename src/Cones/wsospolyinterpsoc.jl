#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

interpolation-based weighted-sum-of-squares (multivariate) polynomial second order cone parametrized by interpolation points ipwt

definition and dual barrier extended from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792
and "Semidefinite characterization of sum-of-squares cones in algebras" by D. Papp and F. Alizadeh, SIAM Journal on Optimization.
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
    tmp2::Matrix{Float64}
    tmp3::Matrix{Float64}
    tmp4::Vector{Matrix{Float64}}
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
        cone.tmp2 = similar(ipwt[1], U, U)
        cone.tmp3 = similar(cone.tmp2)
        cone.tmp4 = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        cone.lambdafact = Vector{CholeskyPivoted{Float64, Matrix{Float64}}}(undef, length(ipwt))
        cone.diffres = DiffResults.HessianResult(cone.g)
        return cone
    end
end

WSOSPolyInterpSOC(R::Int, U::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterpSOC(R, U, ipwt, false)

get_nu(cone::WSOSPolyInterpSOC) = 2 * sum(size(ipwtj, 2) for ipwtj in cone.ipwt)

function set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterpSOC)
    arr .= 0.0
    arr[1:cone.U] .= 1.0
    return arr
end

function lambda(point, cone, j)
    ipwtj = cone.ipwt[j]
    L = size(ipwtj, 2)
    mat = similar(point, L, L)

    # first lambda
    point_pq = point[1:cone.U]
    first_lambda = ipwtj' * Diagonal(point_pq) * ipwtj
    mat = Symmetric(first_lambda)
    first_lambda_inv = inv(Symmetric(first_lambda))

    # minus other lambdas
    uo = cone.U + 1
    for p in 2:cone.R
        point_pq = point[uo:(uo + cone.U - 1)]
        tmp = ipwtj' * Diagonal(point_pq) * ipwtj
        mat -= Symmetric(tmp * first_lambda_inv * tmp')
        uo += cone.U
    end
    return mat
end

function lambda2(point, cone, j)
    ipwtj = cone.ipwt[j]
    L = size(ipwtj, 2)
    mat = similar(point, L, L)

    # first lambda
    point_pq = point[1:cone.U]
    tmp = ipwtj' * Diagonal(point_pq) * ipwtj
    mat = Symmetric(tmp * tmp')

    # minus other lambdas^2
    uo = cone.U + 1
    for p in 2:cone.R
        point_pq = point[uo:(uo + cone.U - 1)]
        tmp = ipwtj' * Diagonal(point_pq) * ipwtj
        mat -= Symmetric(tmp * tmp')
        uo += cone.U
    end
    return mat
end

function linsubmap(
    ipwtj::Matrix{Float64},
    r::Int,
    u::Int,
    v::Int,
    )
    L = size(ipwtj, 2)
    return ipwtj[u, :] * ipwtj[v, :]' * dot(ipwtj[u, :], ipwtj[v, :])
end


function dlambda_dx(cone::WSOSPolyInterpSOC, r::Int, u::Int, j::Int)
    ipwtj = cone.ipwt[j]
    offset = (r - 1) * cone.U
    fact = 1
    if r != 1
        fact = -1
    end
    return fact * sum(cone.point[offset + v] * (linsubmap(ipwtj, r, u, v) + linsubmap(ipwtj, r, v, u)) for v in 1:cone.U)
end

function calchessian(cone::WSOSPolyInterpSOC, r1::Int, r2::Int, u1::Int, u2::Int, j::Int)
    Winv = inv(cone.matfact[j])
    # tmp1 = Winv * dlambda_dx(cone, r2, u2, j) * Winv
    tmp1 = (cone.matfact[j] \ dlambda_dx(cone, r2, u2, j)) * Winv
    tmp2 = dlambda_dx(cone, r1, u1, j)
    tmp3 = sum(tmp1 .* tmp2)
    # @assert isapprox(sum(tmp1 .* tmp2), sum((Winv * dlambda_dx(cone, r1, u1, j) * Winv) .* dlambda_dx(cone, r2, u2, j)))
    if r1 == r2
        fact = (r1 == 1 ? 1 : -1)
        tmp4 = fact * (linsubmap(cone.ipwt[j], r1, u1, u2) + linsubmap(cone.ipwt[j], r1, u2, u1))
        return tmp3 - sum(Winv .* tmp4)
    else
        return tmp3
    end
end



function gradient_try2(cone::WSOSPolyInterpSOC, r::Int, u::Int, j::Int)
    # gr = 0.0
    # offset = (r - 1) * cone.U
    # Winv = Symmetric(inv(cone.matfact[j]))
    # ipwtj = cone.ipwt[j]
    # for v in 1:cone.U
    #     gr -= 2 * cone.point[offset + v] * dot(ipwtj[u, :], ipwtj[v, :]) * (ipwtj[u, :]' * Winv * ipwtj[v, :])
    # end
    # if r != 1
    #     gr *= -1
    # end
    # return gr
    return -sum(inv(cone.matfact[j]) .* dlambda_dx(cone, r, u, j))
end

# function gradient_try3(cone::WSOSPolyInterpSOC, r::Int, j::Int)
#     offset = (r - 1) * cone.U
#     gr = zeros(cone.U)
#     Winv = inv(cone.matfact[j])
#     ipwtj = cone.ipwt[j]
#     for v in 1:cone.U
#         gr -= 2 * cone.point[offset + v] * (ipwtj * ipwtj[v, :]) .* (ipwtj * Winv * ipwtj[v, :])
#     end
#     if r != 1
#         gr *= -1
#     end
#     return gr
# end

function check_in_cone(cone::WSOSPolyInterpSOC)

    # @timeit "build mat" begin
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        L = size(ipwtj, 2)
        mat = cone.mat[j]
        tmp4 = cone.tmp4[j]

        # first lambda
        point_pq = cone.point[1:cone.U]
        @. tmp1j = ipwtj' * point_pq'
        mul!(tmp4, tmp1j, ipwtj)
        lambdafact = cholesky(Symmetric(tmp4, :L), Val(true), check = false)
        if !isposdef(lambdafact)
            return false
        end
        BLAS.syrk!('U', 'N', 1.0, tmp4, 0.0, mat)
        # mat = tmp4 * tmp4'
        # first_lambda_inv = inv(lambdafact)

        # minus other lambdas^2
        uo = cone.U + 1
        for p in 2:cone.R
            point_pq = cone.point[uo:(uo + cone.U - 1)] # TODO prealloc
            @. tmp1j = ipwtj' * point_pq'
            mul!(tmp4, tmp1j, ipwtj)
            BLAS.syrk!('U', 'N', -1.0, tmp4, 1.0, mat)
            # mat -= Symmetric(tmp4 * first_lambda_inv * tmp4')
            uo += cone.U
        end

        cone.matfact[j] = cholesky(Symmetric(mat, :U), Val(true), check = false)
        if !isposdef(cone.matfact[j])
            return false
        end
        # @show Symmetric(mat, :U) - lambda2(cone.point, cone, j)
        @assert isposdef(Symmetric(lambda2(cone.point, cone, j)))
        @assert isposdef(Symmetric(lambda(cone.point, cone, j)))
    end
    # end

    g_try = zeros(cone.U * cone.R)
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        L = size(ipwtj, 2)
        Winv = inv(cone.matfact[j])
        ur = 0
        for r in 1:cone.R, u in 1:cone.U
            ur += 1
            if ur <= cone.U
                fact = 1.0
            else
                fact = -1.0
            end
            for wi in 1:L, wj in 1:L, k in 1:L
                for u2 in 1:cone.U
                    pntu2 = cone.point[(r - 1) * cone.U + u2]
                    if u == u2
                        g_try[ur] -= 2 * Winv[wi, wj] * ipwtj[u, wi] * ipwtj[u, wj] * ipwtj[u, k]^2 * cone.point[ur] * fact
                    else
                        g_try[ur] -= 2 * Winv[wi, wj] * ipwtj[u2, wi] * ipwtj[u, wj] * ipwtj[u2, k] * ipwtj[u, k] * pntu2 * fact
                    end
                end
            end
        end # u
    end # j

    g_try2 = zeros(cone.U * cone.R)
    h_try = zeros(cone.U * cone.R, cone.U * cone.R)
    ur = 0
    for r in 1:cone.R, u in 1:cone.U
        ur += 1
        g_try2[ur] += gradient_try2(cone, r, u, 1)
        ur2 = 0
        for r2 in 1:cone.R, u2 in 1:cone.U
            ur2 += 1
            h_try[ur, ur2] += calchessian(cone, r, r2, u, u2, 1)
        end
    end
    # @show h_try

    cone.g .= 0.0
    cone.H .= 0.0
    for j in eachindex(cone.ipwt)
        cone.diffres = ForwardDiff.hessian!(cone.diffres, x -> -logdet(lambda2(x, cone, j)), cone.point)
        cone.g += DiffResults.gradient(cone.diffres)
        cone.H += DiffResults.hessian(cone.diffres)
    end
    # cone.H .= h_try

    # @show cone.g ./ g_try2
    @show  cone.H ./ h_try
    # @show cone.point
    # @show abs.(cone.H - h_try) ./ (1 .+ min.(cone.H, h_try))
    # @show cone.point
    # @show h_try

    # @timeit "grad hess" begin
    # cone.g .= 0.0
    # cone.H .= 0.0
    # for j in eachindex(cone.ipwt)
    #     # @timeit "W_inv" begin
    #     W_inv_j = inv(cone.matfact[j])
    #     # end
    #
    #     ipwtj = cone.ipwt[j]
    #     tmp1j = cone.tmp1[j]
    #     tmp2 = cone.tmp2
    #     tmp3 = cone.tmp3
    #
    #     L = size(ipwtj, 2)
    #     uo = 0
    #     for p in 1:cone.R
    #         uo += 1
    #         rinds = _blockrange(p, L)
    #         idxs = _blockrange(uo, cone.U)
    #
    #         for i in 1:cone.U
    #             if p == 1
    #                 for r in 1:cone.R
    #                     cone.g[idxs[i]] -= ipwtj[i, :]' * view(W_inv_j, _blockrange(r, L), _blockrange(r, L)) * ipwtj[i, :]
    #                 end
    #             else
    #                 cone.g[idxs[i]] -= 2 * ipwtj[i, :]' * view(W_inv_j, 1:L, rinds) * ipwtj[i, :]
    #             end
    #         end
    #
    #         # @show "hessian"
    #
    #         uo2 = 0
    #         for p2 in 1:cone.R
    #             uo2 += 1
    #             if uo2 < uo
    #                 continue
    #             end
    #             rinds2 = _blockrange(p2, L)
    #             idxs2 = _blockrange(uo2, cone.U)
    #
    #             if p == 1 && p2 == 1
    #                 for r in 1:cone.R, s in 1:cone.R
    #                     cone.H[idxs, idxs2] += (ipwtj * view(W_inv_j, _blockrange(r, L), _blockrange(s, L)) * ipwtj').^2
    #                 end
    #             elseif p == 1 && p2 != 1
    #                 for r in 1:cone.R
    #                     cone.H[idxs, idxs2] += 2 * (ipwtj * view(W_inv_j, _blockrange(1, L), _blockrange(r, L)) * ipwtj') .* (ipwtj * view(W_inv_j, _blockrange(r, L), rinds2) * ipwtj')
    #                 end
    #             elseif p != 1 && p2 == 1
    #                 for r in 1:cone.R
    #                     cone.H[idxs, idxs2] += 2 * (ipwtj * view(W_inv_j, _blockrange(1, L), _blockrange(r, L)) * ipwtj') .* (ipwtj * view(W_inv_j, _blockrange(r, L), rinds) * ipwtj')
    #                 end
    #             else
    #                 cone.H[idxs, idxs2] += 2 * (ipwtj * view(W_inv_j, 1:L, 1:L) * ipwtj') .* (ipwtj * view(W_inv_j, rinds, rinds2) * ipwtj') +
    #                                        2 * (ipwtj * view(W_inv_j, 1:L, rinds) * ipwtj') .* (ipwtj * view(W_inv_j, 1:L, rinds2) * ipwtj')
    #             end
    #         end
    #     end # p
    # end # j
    # end

    return factorize_hess(cone)
end
