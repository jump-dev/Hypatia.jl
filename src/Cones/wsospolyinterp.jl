#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz

interpolation-based weighted-sum-of-squares (multivariate) polynomial cone parametrized by interpolation points ipwt

definition and dual barrier from "Sum-of-squares optimization without semidefinite programming" by D. Papp and S. Yildiz, available at https://arxiv.org/abs/1712.01792

TODO can perform loop for calculating g and H in parallel
TODO maybe can avoid final factorization?
TODO scale the interior direction
=#

mutable struct WSOSPolyInterp <: Cone
    use_dual::Bool
    dim::Int
    ipwt::Vector{Matrix{Float64}}
    point::AbstractVector{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    H2::Matrix{Float64}
    Hi::Matrix{Float64}
    Hitemp::Matrix{Float64}
    F # TODO prealloc
    tmp1::Vector{Matrix{Float64}}
    tmp2::Vector{Matrix{Float64}}
    tmp3::Matrix{Float64}

    function WSOSPolyInterp(dim::Int, ipwt::Vector{Matrix{Float64}}, is_dual::Bool)
        for ipwtj in ipwt
            @assert size(ipwtj, 1) == dim
        end
        cone = new()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.ipwt = ipwt
        cone.g = similar(ipwt[1], dim)
        cone.H = similar(ipwt[1], dim, dim)
        cone.H2 = similar(cone.H)
        cone.Hi = similar(cone.H)
        cone.Hitemp = similar(cone.H)
        cone.tmp1 = [similar(ipwt[1], size(ipwtj, 2), size(ipwtj, 2)) for ipwtj in ipwt]
        cone.tmp2 = [similar(ipwt[1], size(ipwtj, 2), dim) for ipwtj in ipwt]
        cone.tmp3 = similar(ipwt[1], dim, dim)
        return cone
    end
end

WSOSPolyInterp(dim::Int, ipwt::Vector{Matrix{Float64}}) = WSOSPolyInterp(dim, ipwt, false)

get_nu(cone::WSOSPolyInterp) = sum(size(ipwtj, 2) for ipwtj in cone.ipwt)

set_initial_point(arr::AbstractVector{Float64}, cone::WSOSPolyInterp) = (@. arr = 1.0; arr)

function check_in_cone(cone::WSOSPolyInterp)
    @timeit "in scalar cone" begin
    @. cone.g = 0.0
    @. cone.H = 0.0
    @. cone.Hi = 0.0
    @. cone.Hitemp = 0.0
    tmp3 = cone.tmp3

    for j in eachindex(cone.ipwt) # TODO can be done in parallel, but need multiple tmp3s
        ipwtj = cone.ipwt[j]
        tmp1j = cone.tmp1[j]
        tmp2j = cone.tmp2[j]

        # tmp1j = ipwtj'*Diagonal(point)*ipwtj
        # mul!(tmp2j, ipwtj', Diagonal(cone.point)) # TODO dispatches to an extremely inefficient method
        @. tmp2j = ipwtj' * cone.point'
        mul!(tmp1j, tmp2j, ipwtj)

        # pivoted cholesky and triangular solve method
        F = cholesky!(Symmetric(tmp1j, :L), Val(true), check = false)
        if !isposdef(F)
            return false
        end

        tmp2j .= view(ipwtj', F.p, :)
        ldiv!(F.L, tmp2j) # TODO make sure calls best triangular solve
        # mul!(tmp3, tmp2j', tmp2j)
        BLAS.syrk!('U', 'T', 1.0, tmp2j, 0.0, tmp3)

        # inv_Hj = (ipwtj * ipwtj' * Diagonal(cone.point) * ipwtj * ipwtj').^2
        # cone.Hi += inv((ipwtj * inv(ipwtj' * Diagonal(cone.point) * ipwtj) * ipwtj').^2)
        W = ipwtj' * Diagonal(cone.point) * ipwtj
        Winv = inv(W)
        L = size(ipwtj, 2)

        @inbounds for j in eachindex(cone.g)
            cone.g[j] -= tmp3[j, j]
            @inbounds for i in 1:j
                # cone.H[i, j] += abs2(tmp3[i, j])
                cone.H[i, j] += sum(Winv[k, l] * ipwtj[i, k] * ipwtj[j, l] for k in 1:L, l in 1:L)^2
                # cone.Hitemp[i, j] += sum(W[k, l] * ipwtj[i, k] * ipwtj[j, l] for k in 1:L, l in 1:L)^2
            end
        end
    end

    @. cone.H2 = cone.H
    @timeit "cone cholesky" cone.F = cholesky!(Symmetric(cone.H2, :U), Val(true), check = false)
    if !isposdef(cone.F)
        return false
    end
    @timeit "hessian inv" cone.Hi .= inv(cone.F)
    # @show cone.Hi -  (cone.ipwt[1] \ cone.ipwt[1] * Diagonal(cone.point) * cone.ipwt[1]' * cone.ipwt[1]' \ Matrix(I, cone.dim, cone.dim))
    @show Symmetric(cone.Hi, :U) ./ Hinvid(cone.ipwt[1], cone.point)

end

    return true
end

function Harr(ipwtj, x, arr)
    lambda_arr = ipwtj' * Diagonal(arr) * ipwtj
    lambda_x = ipwtj' * Diagonal(x) * ipwtj
    lambda_x_inv = inv(lambda_x)
    return diag(ipwtj * lambda_x_inv * lambda_arr * lambda_x_inv * ipwtj')
    # return diag((lambda_x_inv * lambda_arr * lambda_x_inv) .* (ipwtj' * ipwtj))
end

function Hinvarr(ipwtj, x, arr)
    lambda_arr = ipwtj' * Diagonal(arr) * ipwtj
    lambda_x = ipwtj' * Diagonal(x) * ipwtj
    return diag((ipwtj * ipwtj') \ ipwtj * lambda_x * lambda_arr * lambda_x * ipwtj' * inv(ipwtj * ipwtj'))
end

function Hid(ipwtj, x)
    U = size(ipwtj, 1)
    H = zeros(U, U)
    for i in 1:U
        ei = zeros(U)
        ei[i] = 1.0
        H[:, i] = Harr(ipwtj, x, ei)
    end
    H
end

function Hinvid(ipwtj, x)
    U = size(ipwtj, 1)
    H = zeros(U, U)
    for i in 1:U
        ei = zeros(U)
        ei[i] = 1.0
        H[:, i] = Hinvarr(ipwtj, x, ei)
    end
    H
end

inv_hess_prod!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::WSOSPolyInterp) = mul!(prod, Symmetric(cone.Hi, :U), arr)
