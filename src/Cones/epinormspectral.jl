#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors

epigraph of matrix spectral norm (operator norm associated with standard Euclidean norm; i.e. maximum singular value)
(u in R, W in R^{n,m}) : u >= opnorm(W)
note n <= m is enforced WLOG since opnorm(W) = opnorm(W')
W is vectorized column-by-column (i.e. vec(W) in Julia)

barrier from "Interior-Point Polynomial Algorithms in Convex Programming" by Nesterov & Nemirovskii 1994
-logdet(u*I_n - W*W'/u) - log(u)
= -logdet(u^2*I_n - W*W') + (n - 1) log(u)
=#

mutable struct EpiNormSpectral{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    n::Int
    m::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    hess_prod_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    W::Matrix{T}
    Z::Matrix{T}
    fact_Z
    Zi::Symmetric{T, Matrix{T}}
    ZiW::Matrix{T}
    HuW::Matrix{T}
    Huu::T
    tmpmm::Matrix{T}
    tmpnm::Matrix{T}
    tmpnn::Matrix{T}

    function EpiNormSpectral{T}(
        n::Int,
        m::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert 1 <= n <= m
        dim = n * m + 1
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.n = n
        cone.m = m
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

EpiNormSpectral{T}(n::Int, m::Int) where {T <: Real} = EpiNormSpectral{T}(n, m, false)

reset_data(cone::EpiNormSpectral) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.inv_hess_prod_updated = false)

function setup_data(cone::EpiNormSpectral{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.W = Matrix{T}(undef, cone.n, cone.m)
    cone.Z = Matrix{T}(undef, cone.n, cone.n)
    cone.ZiW = Matrix{T}(undef, cone.n, cone.m)
    cone.HuW = Matrix{T}(undef, cone.n, cone.m)
    cone.tmpmm = Matrix{T}(undef, cone.m, cone.m)
    cone.tmpnm = Matrix{T}(undef, cone.n, cone.m)
    cone.tmpnn = Matrix{T}(undef, cone.n, cone.n)
    return
end

get_nu(cone::EpiNormSpectral) = cone.n + 1

function set_initial_point(arr::AbstractVector, cone::EpiNormSpectral{T}) where {T <: Real}
    arr .= 0
    arr[1] = sqrt(T(get_nu(cone)))
    return arr
end

function update_feas(cone::EpiNormSpectral)
    @assert !cone.feas_updated
    u = cone.point[1]

    if u > 0
        cone.W[:] .= view(cone.point, 2:cone.dim)
        copyto!(cone.Z, abs2(u) * I) # TODO inefficient
        mul!(cone.Z, cone.W, cone.W', -1, true)
        cone.fact_Z = cholesky!(Symmetric(cone.Z, :U), check = false)
        cone.is_feas = isposdef(cone.fact_Z)
    else
        cone.is_feas = false
    end

    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::EpiNormSpectral)
    @assert cone.is_feas
    u = cone.point[1]

    ldiv!(cone.ZiW, cone.fact_Z, cone.W)
    cone.Zi = Symmetric(inv(cone.fact_Z), :U) # TODO only need trace of inverse here, which we can get from the cholesky factor - if cheap, don't do the inverse until needed in the hessian

    cone.grad[1] = -u * tr(cone.Zi)
    cone.grad[2:end] = cone.ZiW
    cone.grad .*= 2
    cone.grad[1] += (cone.n - 1) / u

    cone.grad_updated = true
    return cone.grad
end

function update_hess_prod(cone::EpiNormSpectral)
    @assert cone.grad_updated
    u = cone.point[1]
    HuW = cone.HuW

    copyto!(HuW, cone.ZiW)
    HuW .*= -4u
    ldiv!(cone.fact_Z, HuW)
    cone.Huu = 4 * abs2(u) * sum(abs2, cone.Zi) + (cone.grad[1] - 2 * (cone.n - 1) / u) / u

    cone.hess_prod_updated = true
    return nothing
end

function update_hess(cone::EpiNormSpectral)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    n = cone.n
    m = cone.m
    u = cone.point[1]
    W = cone.W
    Zi = cone.Zi
    ZiW = cone.ZiW
    tmpmm = cone.tmpmm
    H = cone.hess.data

    # H_W_W part
    mul!(tmpmm, W', ZiW) # symmetric, W' * Zi * W
    # TODO parallelize loops
    @inbounds for i in 1:m
        r = 1 + (i - 1) * n
        for j in 1:n
            r2 = r + j
            @. @views H[r2, r .+ (j:n)] = Zi[j:n, j] * tmpmm[i, i] + ZiW[j:n, i] * ZiW[j, i] + Zi[j, j:n]
            c2 = r + n
            @inbounds for k in (i + 1):m
                @. @views H[r2, c2 .+ (1:n)] = Zi[1:n, j] * tmpmm[i, k] + ZiW[1:n, i] * ZiW[j, k]
                c2 += n
            end
        end
    end
    H .*= 2

    # H_u_W and H_u_u parts
    H[1, 2:end] .= vec(cone.HuW)
    H[1, 1] = cone.Huu


    # # TODO try replacing Zi with Z in order to derive an inv hess
    # # H_W_W part
    # Hi = similar(H)
    # Hi .= 0
    # Z = Symmetric(abs2(u) * I - W * W')
    # ZW = Z * W
    # tmpmm = W' * ZW
    #
    # @inbounds for i in 1:m
    #     r = 1 + (i - 1) * n
    #     for j in 1:n
    #         r2 = r + j
    #         @. @views Hi[r2, r .+ (j:n)] = Z[j:n, j] * tmpmm[i, i] + ZW[j:n, i] * ZW[j, i] + Z[j, j:n]
    #         c2 = r + n
    #         @inbounds for k in (i + 1):m
    #             @. @views Hi[r2, c2 .+ (1:n)] = Z[1:n, j] * tmpmm[i, k] + ZW[1:n, i] * ZW[j, k]
    #             c2 += n
    #         end
    #     end
    # end
    # Hi ./= 2
    #
    # # H_u_W and H_u_u parts
    # Hi[1, 2:end] .= vec(u * W)
    #
    #
    # Hi_try = Symmetric(Hi, :U)
    # Hi_true = inv(cholesky(cone.hess))
    # println(round.(Hi_try[:, 2:end], digits=10))
    # println(round.(Hi_true[:, 2:end], digits=10))
    # println(round.((Hi_try - Hi_true)[:, 2:end], digits=10))
    # println()




    cone.hess_updated = true
    return cone.hess
end

# function update_inv_hess(cone::EpiNormSpectral)
#     if !cone.hess_prod_updated
#         update_hess_prod(cone)
#     end
#     n = cone.n
#     m = cone.m
#     u = cone.point[1]
#     W = cone.W
#     Z = cone.Z
#     tmpmm = cone.tmpmm
#     H = cone.inv_hess.data
#
#     idx = 1
#     for j in 1:m, i in 1:n
#         idx2 = 1
#         for j2 in 1:m, i2 in 1:n
#             H[idx, idx2] = W[i, j] * W[i2, j2]
#             if j == j2
#                 H[idx, idx2] += Z[i, i2]
#             end
#             idx2 += 1
#         end
#         idx += 1
#     end
#
#     cone.hess_updated = true
#     return cone.hess
# end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormSpectral)
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    u = cone.point[1]
    W = cone.W
    tmpnm = cone.tmpnm
    tmpnn = cone.tmpnn

    @inbounds for j in 1:size(prod, 2)
        arr_1j = arr[1, j]
        tmpnm[:] .= @view arr[2:end, j]

        prod[1, j] = cone.Huu * arr_1j + dot(cone.HuW, tmpnm)

        # prod_2j = 2 * cone.fact_Z \ (((tmpnm * W' + W * tmpnm' - (2 * u * arr_1j) * I) / cone.fact_Z) * W + tmpnm)
        mul!(tmpnn, tmpnm, W')
        @inbounds for j in 1:cone.n
            @. @views tmpnn[1:(j - 1), j] += tmpnn[j, 1:(j - 1)]
            tmpnn[j, j] -= u * arr_1j
            tmpnn[j, j] *= 2
        end
        mul!(tmpnm, Symmetric(tmpnn, :U), cone.ZiW, 2, 2)
        ldiv!(cone.fact_Z, tmpnm)
        prod[2:end, j] .= vec(tmpnm)
    end

    return prod
end

# TODO fix
# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormSpectral)
#     # if !cone.hess_prod_updated
#     #     update_hess_prod(cone)
#     # end
#     u = cone.point[1]
#     W = cone.W
#     tmpnm = cone.tmpnm
#     # tmpnn = cone.tmpnn
#
#     Z = abs2(u) * I - W * W' # TODO use previous, but need to make cholesky not overwrite it
#
#     @inbounds for j in 1:size(prod, 2)
#         arr_1j = arr[1, j]
#         tmpnm[:] .= @view arr[2:end, j]
#
#         pa = (tmpnm * W' + W * tmpnm') / 2 + (cone.n * u * arr_1j) * I
#         prod[1, j] = tr(pa) * u - tr(Z) * arr_1j / 2
#
#         tmpnm2 = pa * W + Z / 2 * tmpnm
#         prod[2:end, j] .= vec(tmpnm2)
#     end
#
#     return prod
# end

# function inv_hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::EpiNormSpectral)
#     if !cone.hess_prod_updated
#         update_hess_prod(cone)
#     end
#     u = cone.point[1]
#     W = cone.W
#     tmpnm = cone.tmpnm
#     tmpnn = cone.tmpnn
#
#     Z = abs2(u) * I - W * W'
#
#     @inbounds for j in 1:size(prod, 2)
#         arr_1j = arr[1, j]
#         tmpnm[:] .= @view arr[2:end, j]
#
#         prod[1, j] = cone.Huu * arr_1j + dot(cone.HuW, tmpnm)
#
#         # prod_2j = 2 * cone.fact_Z \ (((tmpnm * W' + W * tmpnm' - (2 * u * arr_1j) * I) / cone.fact_Z) * W + tmpnm)
#         mul!(tmpnn, tmpnm, -W')
#         @inbounds for j in 1:cone.n
#             @. @views tmpnn[1:(j - 1), j] += tmpnn[j, 1:(j - 1)]
#             tmpnn[j, j] -= u * arr_1j
#             tmpnn[j, j] *= 2
#         end
#         tmpnm += Symmetric(tmpnn, :U) * Z * -W
#         tmpnm /= 2
#         tmpnm .= Z * tmpnm
#         # mul!(tmpnm, Symmetric(tmpnn, :U), cone.ZiW, 2, 2)
#         # ldiv!(cone.fact_Z, tmpnm)
#         prod[2:end, j] .= vec(tmpnm)
#     end
#
#     return prod
# end
