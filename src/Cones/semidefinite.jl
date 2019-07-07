#=
Copyright 2018, Chris Coey and contributors

TODO describe hermitian complex PSD cone
on-diagonal (real) elements have one slot in the vector and below diagonal (complex) elements have two consecutive slots in the vector

row-wise lower triangle (svec space) of positive semidefinite matrix cone
(smat space) W \in S^n : 0 >= eigmin(W)
(see equivalent MathOptInterface PositiveSemidefiniteConeTriangle definition)

barrier from "Self-Scaled Barriers and Interior-Point Methods for Convex Programming" by Nesterov & Todd
-logdet(W)

TODO
- eliminate allocations for inverse-finding
- eliminate redundant svec_to_smat calls
=#

mutable struct PosSemidef{T <: HypReal, R <: HypRealOrComplex{T}} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int

    point::AbstractVector{T}
    g::Vector{T}
    H::Matrix{T}
    Hi::Matrix{T}
    mat::Matrix{R}

    function PosSemidef{T, R}(dim::Int, is_dual::Bool) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
        cone = new{T, R}()
        cone.use_dual = is_dual
        cone.dim = dim # real vector dimension
        if R <: Complex
            side = isqrt(dim) # real lower triangle and imaginary under diagonal
            @assert side^2 == dim
        else
            side = round(Int, sqrt(0.25 + 2.0 * dim) - 0.5) # real lower triangle
            @assert side * (side + 1) == 2 * dim
        end
        cone.side = side
        return cone
    end
end

PosSemidef{T, R}(dim::Int) where {R <: HypRealOrComplex{T}} where {T <: HypReal} = PosSemidef{T, R}(dim, false)

function setup_data(cone::PosSemidef{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
    dim = cone.dim
    cone.g = Vector{T}(undef, dim)
    cone.H = zeros(T, dim, dim)
    cone.Hi = zeros(T, dim, dim)
    cone.mat = Matrix{R}(undef, cone.side, cone.side)
    return
end

get_nu(cone::PosSemidef) = cone.side

function set_initial_point(arr::AbstractVector{T}, cone::PosSemidef{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
    incr_off = (R <: Complex) ? 2 : 1
    arr .= 0.0
    k = 1
    for i in 1:cone.side, j in 1:i
        if i == j # on diagonal
            arr[k] = 1.0
            k += 1
        else # off diagonal
            k += incr_off
        end
    end
    return arr
end

function check_in_cone(cone::PosSemidef{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal}
    Hypatia.TimerOutputs.@timeit Hypatia.to "all" begin
    mat = cone.mat
    Hypatia.TimerOutputs.@timeit Hypatia.to "smat_pt" svec_to_smat!(mat, cone.point)
    Hypatia.TimerOutputs.@timeit Hypatia.to "chol" F = hyp_chol!(Hermitian(mat))
    if !isposdef(F)
        return false
    end

    Hypatia.TimerOutputs.@timeit Hypatia.to "inversion" inv_mat = Hermitian(inv(F)) # TODO eliminate allocs
    Hypatia.TimerOutputs.@timeit Hypatia.to "smat_to_svec" smat_to_svec!(cone.g, transpose(inv_mat)) # TODO avoid doing this twice
    cone.g .*= -1

    # set upper triangles of hessian and inverse hessian
    Hypatia.TimerOutputs.@timeit Hypatia.to "svec_to_smat" svec_to_smat!(mat, cone.point)
    H = cone.H
    Hi = cone.Hi
    rt2 = sqrt(T(2))

    # TODO refactor
    Hypatia.TimerOutputs.@timeit Hypatia.to "if" if R <: Complex
        k = 1
        for i in 1:cone.side, j in 1:i
            k2 = 1
            if i == j
                for i2 in 1:cone.side, j2 in 1:i2
                    if i2 == j2
                        H[k2, k] = abs2(inv_mat[i2, i])
                        Hi[k2, k] = abs2(mat[i2, i])
                        k2 += 1
                    else
                        c = rt2 * conj(inv_mat[i2, i]) * inv_mat[j2, j]
                        ci = rt2 * conj(mat[i2, i]) * mat[j2, j]
                        H[k2, k] = real(c)
                        Hi[k2, k] = real(ci)
                        k2 += 1
                        H[k2, k] = imag(c)
                        Hi[k2, k] = imag(ci)
                        k2 += 1
                    end
                    if k2 > k
                        break
                    end
                end
                k += 1
            else
                for i2 in 1:cone.side, j2 in 1:i2
                    if i2 == j2 # TODO try to merge with other XOR condition above
                        c = rt2 * inv_mat[i2, i] * conj(inv_mat[j2, j])
                        ci = rt2 * mat[i2, i] * (j2 >= j ? mat[j2, j] : conj(mat[j2, j]))
                        H[k2, k] = real(c)
                        Hi[k2, k] = real(ci)
                        H[k2, k + 1] = imag(c)
                        Hi[k2, k + 1] = imag(ci)
                        k2 += 1
                    else
                        c = inv_mat[i2, i] * conj(inv_mat[j2, j]) + inv_mat[j2, i] * conj(inv_mat[i2, j])
                        ci = mat[i2, i] * (j2 >= j ? mat[j2, j] : conj(mat[j2, j])) + mat[j2, i] * (i2 >= j ? mat[i2, j] : conj(mat[i2, j]))
                        c2 = conj(inv_mat[i2, i]) * inv_mat[j2, j] - conj(inv_mat[j2, i]) * inv_mat[i2, j]
                        c2i = conj(mat[i2, i]) * (j2 < j ? mat[j2, j] : conj(mat[j2, j])) - conj(mat[j2, i]) * (i2 < j ? mat[i2, j] : conj(mat[i2, j]))
                        H[k2, k] = real(c)
                        Hi[k2, k] = real(ci)
                        H[k2, k + 1] = imag(c)
                        Hi[k2, k + 1] = imag(ci)
                        k2 += 1
                        H[k2, k] = imag(c2)
                        Hi[k2, k] = imag(c2i)
                        H[k2, k + 1] = real(c2)
                        Hi[k2, k + 1] = real(c2i)
                        k2 += 1
                    end
                    if k2 > k
                        break
                    end
                end
                k += 2
            end
        end
    else
        Hypatia.TimerOutputs.@timeit Hypatia.to "semidefbar" begin
        k = 1
        Hypatia.TimerOutputs.@timeit Hypatia.to "for1" for i in 1:cone.side, j in 1:i
            k2 = 1
            Hypatia.TimerOutputs.@timeit Hypatia.to "for2" for i2 in 1:cone.side, j2 in 1:i2
                Hypatia.TimerOutputs.@timeit Hypatia.to "ifa" if (i == j) && (i2 == j2)
                    Hypatia.TimerOutputs.@timeit Hypatia.to "abs21" H[k2, k] = abs2(inv_mat[i2, i])
                    Hypatia.TimerOutputs.@timeit Hypatia.to "abs22" Hi[k2, k] = abs2(mat[i2, i])
                elseif (i != j) && (i2 != j2)
                    Hypatia.TimerOutputs.@timeit Hypatia.to "1" H[k2, k] = inv_mat[i2, i] * inv_mat[j, j2] + inv_mat[j2, i] * inv_mat[j, i2]
                    Hypatia.TimerOutputs.@timeit Hypatia.to "2" Hi[k2, k] = mat[i2, i] * mat[j, j2] + mat[j2, i] * mat[j, i2]
                else
                    Hypatia.TimerOutputs.@timeit Hypatia.to "3" H[k2, k] = rt2 * inv_mat[i2, i] * inv_mat[j, j2]
                    Hypatia.TimerOutputs.@timeit Hypatia.to "4" Hi[k2, k] = rt2 * mat[i2, i] * mat[j, j2]
                end
                Hypatia.TimerOutputs.@timeit Hypatia.to "if2" if k2 == k
                    break
                end
                k2 += 1
            end
            k += 1
        end
        end # time
    end

    return true
    end # time
end

inv_hess(cone::PosSemidef) = Symmetric(cone.Hi, :U)

inv_hess_prod!(prod::AbstractVecOrMat{T}, arr::AbstractVecOrMat{T}, cone::PosSemidef{T, R}) where {R <: HypRealOrComplex{T}} where {T <: HypReal} = mul!(prod, Symmetric(cone.Hi, :U), arr)
