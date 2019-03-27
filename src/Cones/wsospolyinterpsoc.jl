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
    tmp3::Matrix{Float64} # unused
    tmp4::Vector{Matrix{Float64}}
    tmp5::Vector{Matrix{Float64}}
    lambda::Vector{Vector{Matrix{Float64}}}
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
        cone.tmp5 = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        cone.lambda = [Vector{Matrix{Float64}}(undef, R) for ipwtj in ipwt]
        for j in eachindex(ipwt), r in 1:R
            cone.lambda[j][r] = similar(ipwt[1], size(ipwt[j], 2), size(ipwt[j], 2))
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

function getlambda(point, cone, j)
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
        tmp = Symmetric(ipwtj' * Diagonal(point_pq) * ipwtj)
        mat -= Symmetric(tmp * first_lambda_inv * tmp')
        uo += cone.U
    end
    return Symmetric(mat, :U)
end

function rank_one_inv_update(cone::WSOSPolyInterpSOC, r1::Int, r2::Int, j::Int)
    # TODO change cholesky in incone to first_lambda  / other lambdas and cache it since it'll be reused here
    L = size(cone.ipwt[j], 2)
    lambda_inv = inv(cone.lambdafact[j])
    # @assert r1 >= r2
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

function mat_inv(cone::WSOSPolyInterpSOC, r1::Int, r2::Int, j::Int)
    ret = rank_one_inv_update(cone, r1, r2, j)
    if (r1 == r2) && (r1 != 1)
        ret += Symmetric(inv(cone.lambdafact[j]))
    end
    return ret
end

# TODO figure out how to make feasible dual points and move out into test folder
# TODO think about if the inversion is legitimate if one of the lambdas is zero (or maybe indefinite in this analogy?)
function inversion_test(cone, L, R, j)
    arrow_mat = kron(Matrix{Float64}(I, R, R), cone.lambda[j][1])
    for r in 2:R
        arrow_mat[((r - 1) * L + 1):(r * L), 1:L] .= cone.lambda[j][r]
    end
    arrow_mat_inv = zeros(R * L, R * L)
    for r1 in 1:R, r2 in 1:R
        arrow_mat_inv[((r1 - 1) * L + 1):(r1 * L), ((r2 - 1) * L + 1):(r2 * L)] = mat_inv(cone, r1, r2, j)
    end
    @assert arrow_mat_inv * Symmetric(arrow_mat, :L) ≈ I # TODO move out to a test
    return true
end


function check_in_cone(cone::WSOSPolyInterpSOC)

    # @timeit "build mat" begin
    for j in eachindex(cone.ipwt)
        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        lambda = cone.lambda[j]
        L = size(ipwtj, 2)
        mat = cone.mat[j]
        tmp4 = cone.tmp4[j]
        tmp5 = cone.tmp5[j] # TODO, unneeded

        lambdafact = cone.lambdafact

        # first lambda
        point_pq = cone.point[1:cone.U]
        @. tmp1j = ipwtj' * point_pq'
        mul!(mat, tmp1j, ipwtj)
        lambda[1] .= mat
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
            mul!(tmp4, tmp1j, ipwtj)
            lambda[p] .= tmp4
            # update mat
            tmp5 .= view(tmp4, lambdafact[j].p, :)
            ldiv!(lambdafact[j].L, tmp5)
            BLAS.syrk!('U', 'T', 1.0, tmp5, 0.0, tmp4)
            mat .-= Symmetric(tmp4, :U)
            uo += cone.U
        end

        cone.matfact[j] = cholesky(Symmetric(mat, :U), Val(true), check = false)
        if !isposdef(cone.matfact[j])
            return false
        end
        @assert inversion_test(cone, L, cone.R, j)
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

        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        tmp2j = cone.tmp2[j]
        tmp3 = cone.tmp3

        L = size(ipwtj, 2)
        uo = 0

        # part of gradient/hessian when p=1 TODO rest of the blocks should also look like this
        tmp2j .= view(ipwtj', cone.lambdafact[j].p, :)
        ldiv!(cone.lambdafact[j].L, tmp2j)
        BLAS.syrk!('U', 'T', 1.0, tmp2j, 0.0, tmp3)

        for p in 1:cone.R
            uo += 1
            idxs = _blockrange(uo, cone.U)
            for i in 1:cone.U
                if p == 1
                    cone.g[idxs[i]] += tmp3[i, i] * (cone.R - 1)
                    for r in 1:cone.R
                        cone.g[idxs[i]] -= ipwtj[i, :]' * Winv(r, r) * ipwtj[i, :]
                    end
                else
                    cone.g[idxs[i]] -= ipwtj[i, :]' * Winv(p, 1) * ipwtj[i, :] + ipwtj[i, :]' * Winv(1, p) * ipwtj[i, :]
                end
            end

            uo2 = 0
            for p2 in 1:cone.R
                uo2 += 1
                if uo2 < uo
                    continue
                end
                idxs2 = _blockrange(uo2, cone.U)

                if p == 1 && p2 == 1
                    cone.H[idxs, idxs2] -= tmp3.^2 * (cone.R - 1)
                    for r in 1:cone.R, s in 1:cone.R
                        cone.H[idxs, idxs2] += (ipwtj * Winv(r, s) * ipwtj').^2
                    end
                elseif p == 1 && p2 != 1
                    for r in 1:cone.R
                        cone.H[idxs, idxs2] += (ipwtj * Winv(r, 1) * ipwtj') .* (ipwtj * Winv(p2, r) * ipwtj') +
                                               (ipwtj * Winv(r, p2) * ipwtj') .* (ipwtj * Winv(1, r) * ipwtj')
                    end
                else
                    cone.H[idxs, idxs2] += (ipwtj * Winv(1, 1) * ipwtj') .* (ipwtj * Winv(p, p2) * ipwtj') +
                                           (ipwtj * Winv(1, 1) * ipwtj') .* (ipwtj * Winv(p2, p) * ipwtj') +
                                           (ipwtj * Winv(1, p) * ipwtj') .* (ipwtj * Winv(1, p2) * ipwtj') +
                                           (ipwtj * Winv(p, 1) * ipwtj') .* (ipwtj * Winv(p2, 1) * ipwtj')
                end
            end
        end # p
    end # j
    # end
    # @show fdg ./ cone.g
    @show fdh ./ Symmetric(cone.H, :U)
    @show fdh - Symmetric(cone.H, :U)
    @show fdh
    # @show Symmetric(cone.H, :U)
    # @show Symmetric(cone.H, :U)

    return factorize_hess(cone)
end
