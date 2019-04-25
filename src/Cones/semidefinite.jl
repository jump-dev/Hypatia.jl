#=
Copyright 2018, Chris Coey and contributors

row-wise lower triangle (svec space) of positive semidefinite matrix cone
(smat space) W \in S^n : 0 >= eigmin(W)
(see equivalent MathOptInterface PositiveSemidefiniteConeTriangle definition)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-logdet(W)

TODO eliminate allocations for inverse-finding
=#

mutable struct PosSemidef <: Cone
    use_dual::Bool
    dim::Int
    side::Int
    point::AbstractVector{Float64}
    mat::Matrix{Float64}
    g::Vector{Float64}
    H::Matrix{Float64}
    Hi::Matrix{Float64}

    function PosSemidef(dim::Int, is_dual::Bool)
        cone = new()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.side = round(Int, sqrt(0.25 + 2.0 * dim) - 0.5)
        cone.mat = Matrix{Float64}(undef, cone.side, cone.side)
        cone.g = Vector{Float64}(undef, dim)
        cone.H = zeros(dim, dim)
        cone.Hi = copy(cone.H)
        return cone
    end
end

PosSemidef(dim::Int) = PosSemidef(dim, false)

get_nu(cone::PosSemidef) = cone.side

function set_initial_point(arr::AbstractVector{Float64}, cone::PosSemidef)
    k = 1
    for i in 1:cone.side, j in 1:i
        if i == j
            arr[k] = 1.0
        else
            arr[k] = 0.0
        end
        k += 1
    end
    return arr
end

function check_in_cone(cone::PosSemidef)
    mat = cone.mat
    svec_to_smat!(mat, cone.point)
    F = cholesky!(Symmetric(mat), Val(true), check = false)
    if !isposdef(F)
        return false
    end

    inv_mat = inv(F) # TODO eliminate allocs
    smat_to_svec!(cone.g, inv_mat)
    cone.g .*= -1.0

    # set upper triangles of hessian and inverse hessian
    svec_to_smat!(mat, cone.point)
    H = cone.H
    Hi = cone.Hi

    # TODO remove ifs

    k = 1
    for i in 1:cone.side
        for j in 1:(i - 1)
            k2 = 1
            for i2 in 1:cone.side
                for j2 in 1:(i2 - 1)
                    # i < j and i2 < j2
                    H[k2, k] = inv_mat[i2, i] * inv_mat[j, j2] + inv_mat[j2, i] * inv_mat[j, i2]
                    Hi[k2, k] = mat[i2, i] * mat[j, j2] + mat[j2, i] * mat[j, i2]
                    k2 += 1
                end
                # i < j and i2 == j2
                H[k2, k] = rt2 * inv_mat[i2, i] * inv_mat[j, i2]
                Hi[k2, k] = rt2 * mat[i2, i] * mat[j, i2]
                k2 += 1
            end
            k += 1
        end

        k2 = 1
        for i2 in 1:cone.side
            for j2 in 1:(i2 - 1)
                # i == j, j2 < i2
                H[k2, k] = rt2 * inv_mat[i2, i] * inv_mat[i, j2]
                Hi[k2, k] = rt2 * mat[i2, i] * mat[i, j2]
                k2 += 1
            end
            # i == j, i2 == j2
            H[k2, k] = abs2(inv_mat[i2, i])
            Hi[k2, k] = abs2(mat[i2, i])
            k2 += 1
        end

        k += 1

    end

    return true
end

inv_hess(cone::PosSemidef) = Symmetric(cone.Hi, :U)

inv_hess_prod!(prod::AbstractArray{Float64}, arr::AbstractArray{Float64}, cone::PosSemidef) = mul!(prod, Symmetric(cone.Hi, :U), arr)
