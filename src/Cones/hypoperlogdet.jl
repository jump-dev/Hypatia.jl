#=
Copyright 2018, Chris Coey and contributors

(closure of) hypograph of perspective of (natural) log of determinant of a (row-wise lower triangle i.e. svec space) symmetric positive define matrix
(smat space) (u in R, v in R_+, w in S_+) : u <= v*logdet(W/v)
(see equivalent MathOptInterface LogDetConeConeTriangle definition)

barrier (guessed, based on analogy to hypoperlog barrier)
-log(v*logdet(W/v) - u) - logdet(W) - log(v)

TODO remove allocations
=#

mutable struct HypoPerLogdet{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    point::AbstractVector{T}

    is_feas::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    mat::Matrix{T}
    F

    function HypoPerLogdet{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.side = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
        return cone
    end
end

HypoPerLogdet{T}(dim::Int) where {T <: HypReal} = HypoPerLogdet{T}(dim, false)

function setup_data(cone::HypoPerLogdet{T}) where {T <: HypReal}
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.side = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
    cone.mat = Matrix{T}(undef, cone.side, cone.side)
    return
end

get_nu(cone::HypoPerLogdet) = cone.side + 2

function set_initial_point(arr::AbstractVector, cone::HypoPerLogdet)
    arr[1] = -1
    arr[2] = 1
    k = 3
    for i in 1:cone.side, j in 1:i
        arr[k] = (i == j) ? 1 : 0
        k += 1
    end
    return arr
end

reset_data(cone::HypoPerLogdet) = (cone.is_feas = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO remove allocs
function update_feas(cone::HypoPerLogdet)
    @assert !cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    if v > 0
        svec_to_smat!(cone.mat, view(cone.point, 3:cone.dim))
        fact_mat = hyp_chol!(Symmetric(cone.mat))
        if !isposdef(fact_mat)
            return false
        end
        ldW = logdet(fact_mat)
        if u >= v * (ldW - cone.side * log(v))
            return false
        end
        cone.is_feas = true
    end
    return cone.is_feas
end

function update_grad(cone::HypoPerLogdet)
    @assert cone.is_feas

    L = ldW - cone.side * log(v)
    z = v * L - u

    Wi = Symmetric(inv(F))
    n = cone.side
    dim = cone.dim
    vzi = v / z

    cone.g[1] = inv(z)
    cone.g[2] = (T(n) - L) / z - inv(v)
    gwmat = -Wi * (one(T) + vzi)
    smat_to_svec!(view(cone.g, 3:dim), gwmat)


    cone.grad_updated = true
    return cone.grad
end

# TODO only work with upper triangle
function update_hess(cone::HypoPerLogdet)
    @assert cone.grad_updated

    cone.H[1, 1] = inv(z) / z
    cone.H[1, 2] = (T(n) - L) / z / z
    Huwmat = -vzi * Wi / z
    smat_to_svec!(view(cone.H, 1, 3:dim), Huwmat)

    cone.H[2, 2] = abs2(T(-n) + L) / z / z + T(n) / (v * z) + inv(v) / v
    Hvwmat = ((T(-n) + L) * vzi - one(T)) * Wi / z
    smat_to_svec!(view(cone.H, 2, 3:dim), Hvwmat)

    rt2 = sqrt(T(2))

    k = 3
    for i in 1:n, j in 1:i
        k2 = 3
        for i2 in 1:n, j2 in 1:i2
            if (i == j) && (i2 == j2)
                cone.H[k2, k] = abs2(Wi[i2, i]) * (vzi + one(T)) + Wi[i, i] * Wi[i2, i2] * abs2(vzi)
            elseif (i != j) && (i2 != j2)
                cone.H[k2, k] = (Wi[i2, i] * Wi[j, j2] + Wi[j2, i] * Wi[j, i2]) * (vzi + one(T)) + 2 * Wi[i, j] * Wi[i2, j2] * abs2(vzi)
            else
                cone.H[k2, k] = rt2 * (Wi[i2, i] * Wi[j, j2] * (vzi + one(T)) + Wi[i, j] * Wi[i2, j2] * abs2(vzi))
            end
            if k2 == k
                break
            end
            k2 += 1
        end
        k += 1
    end


    cone.hess_updated = true
    return cone.hess
end


#
# # L = logdet(W / v)
# L = ldW - cone.side * log(v)
# z = v * L - u
#
# Wi = Symmetric(inv(F))
# n = cone.side
# dim = cone.dim
# vzi = v / z
#
# cone.g[1] = inv(z)
# cone.g[2] = (T(n) - L) / z - inv(v)
# gwmat = -Wi * (one(T) + vzi)
# smat_to_svec!(view(cone.g, 3:dim), gwmat)
#
# cone.H[1, 1] = inv(z) / z
# cone.H[1, 2] = (T(n) - L) / z / z
# Huwmat = -vzi * Wi / z
# smat_to_svec!(view(cone.H, 1, 3:dim), Huwmat)
#
# cone.H[2, 2] = abs2(T(-n) + L) / z / z + T(n) / (v * z) + inv(v) / v
# Hvwmat = ((T(-n) + L) * vzi - one(T)) * Wi / z
# smat_to_svec!(view(cone.H, 2, 3:dim), Hvwmat)
#
# rt2 = sqrt(T(2))
#
# k = 3
# for i in 1:n, j in 1:i
#     k2 = 3
#     for i2 in 1:n, j2 in 1:i2
#         if (i == j) && (i2 == j2)
#             cone.H[k2, k] = abs2(Wi[i2, i]) * (vzi + one(T)) + Wi[i, i] * Wi[i2, i2] * abs2(vzi)
#         elseif (i != j) && (i2 != j2)
#             cone.H[k2, k] = (Wi[i2, i] * Wi[j, j2] + Wi[j2, i] * Wi[j, i2]) * (vzi + one(T)) + 2 * Wi[i, j] * Wi[i2, j2] * abs2(vzi)
#         else
#             cone.H[k2, k] = rt2 * (Wi[i2, i] * Wi[j, j2] * (vzi + one(T)) + Wi[i, j] * Wi[i2, j2] * abs2(vzi))
#         end
#         if k2 == k
#             break
#         end
#         k2 += 1
#     end
#     k += 1
# end
