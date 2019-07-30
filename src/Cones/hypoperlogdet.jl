#=
Copyright 2018, Chris Coey and contributors

(closure of) hypograph of perspective of (natural) log of determinant of a (row-wise lower triangle) symmetric positive define matrix
(mat space) (u in R, v in R_+, w in S_+) : u <= v*logdet(W/v)
(see equivalent MathOptInterface LogDetConeConeTriangle definition)

barrier (guessed, based on analogy to hypoperlog barrier)
-log(v*logdet(W/v) - u) - logdet(W) - log(v)

TODO remove allocations
TODO get rid of cone.vecn used by wrapping muls in the Hessian with BLAS functions
=#

mutable struct HypoPerLogdet{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    point::AbstractVector{T}

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

    mat::Matrix{T}
    mat2::Matrix{T}
    mat3::Matrix{T}
    vecn::Vector{T}
    fact_mat
    ldWv::T
    z::T
    Wi::Matrix{T}
    nLz::T
    ldWvuv::T
    vzip1::T
    Wivzi::Matrix{T}
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact # TODO prealloc

    function HypoPerLogdet{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.side = round(Int, sqrt(0.25 + 2 * (dim - 2)) - 0.5)
        return cone
    end
end

HypoPerLogdet{T}(dim::Int) where {T <: HypReal} = HypoPerLogdet{T}(dim, false)

reset_data(cone::HypoPerLogdet) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = cone.hess_prod_updated = cone.inv_hess_prod_updated = false)

function setup_data(cone::HypoPerLogdet{T}) where {T <: HypReal}
    reset_data(cone)
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.mat = Matrix{T}(undef, cone.side, cone.side)
    cone.mat2 = similar(cone.mat)
    cone.mat3 = similar(cone.mat)
    cone.vecn = Vector{T}(undef, cone.dim - 2)
    cone.Wivzi = similar(cone.mat)
    return
end

get_nu(cone::HypoPerLogdet) = cone.side + 2

function set_initial_point(arr::AbstractVector, cone::HypoPerLogdet)
    arr .= 0
    arr[1] = -1
    arr[2] = 1
    k = 3
    for i in 1:cone.side
        arr[k] = 1
        k += i + 1
    end
    return arr
end

# TODO remove allocs
function update_feas(cone::HypoPerLogdet)
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    if v > 0
        vec_to_mat_U!(cone.mat, view(cone.point, 3:cone.dim))
        cone.fact_mat = hyp_chol!(Symmetric(cone.mat, :U))
        if isposdef(cone.fact_mat)
            cone.ldWv = logdet(cone.fact_mat) - cone.side * log(v)
            cone.z = v * cone.ldWv - u
            cone.is_feas = (cone.z > 0)
        else
            cone.is_feas = false
        end
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoPerLogdet)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    cone.Wi = inv(cone.fact_mat)
    cone.nLz = (cone.side - cone.ldWv) / cone.z
    cone.ldWvuv = cone.ldWv - u / v
    cone.vzip1 = 1 + inv(cone.ldWvuv)
    cone.grad[1] = inv(cone.z)
    cone.grad[2] = cone.nLz - inv(v)
    gend = view(cone.grad, 3:cone.dim)
    mat_U_to_vec_scaled!(gend, cone.Wi)
    gend .*= -cone.vzip1
    cone.grad_updated = true
    return cone.grad
end

# TODO only work with upper triangle
function update_hess(cone::HypoPerLogdet)
    if !cone.hess_prod_updated
        # will fill in first two rows of the Hessian and compute Wivzi
        update_hess_prod(cone)
    end
    Wi = cone.Wi
    Wivzi = cone.Wivzi

    k1 = 3
    for i in 1:cone.side, j in 1:i
        k2 = 3
        for i2 in 1:cone.side, j2 in 1:i2
            if (i == j) && (i2 == j2)
                cone.hess.data[k2, k1] = abs2(Wi[i2, i]) * cone.vzip1 + Wivzi[i, i] * Wivzi[i2, i2]
            elseif (i != j) && (i2 != j2)
                cone.hess.data[k2, k1] = 2 * (Wi[i2, i] * Wi[j, j2] + Wi[j2, i] * Wi[j, i2]) * cone.vzip1 + 4 * Wivzi[i, j] * Wivzi[i2, j2]
            else
                cone.hess.data[k2, k1] = 2 * (Wi[i2, i] * Wi[j, j2] * cone.vzip1 + Wivzi[i, j] * Wivzi[i2, j2])
            end
            if k2 == k1
                break
            end
            k2 += 1
        end
        k1 += 1
    end
    cone.hess_updated = true
    return cone.hess
end

# updates first two rows of the Hessian
function update_hess_prod(cone::HypoPerLogdet)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    z = cone.z
    Wi = cone.Wi
    Wivzi = cone.Wivzi
    @. Wivzi = Wi / cone.ldWvuv

    cone.hess.data[1, 1] = inv(z) / z
    cone.hess.data[1, 2] = cone.nLz / z
    h1end = view(cone.hess.data, 1, 3:cone.dim)
    mat_U_to_vec_scaled!(h1end, Wivzi)
    h1end ./= -z
    cone.hess.data[2, 2] = abs2(cone.nLz) + (cone.side / z + inv(v)) / v
    h2end = view(cone.hess.data, 2, 3:cone.dim)
    mat_U_to_vec_scaled!(h2end, Wi)
    h2end .*= ((cone.ldWv - cone.side) / cone.ldWvuv - 1) / z
    cone.hess_prod_updated = true
    return nothing
end

function hess_prod!(prod::AbstractVecOrMat, arr::AbstractVecOrMat, cone::HypoPerLogdet{T}) where {T <: HypReal}
    if !cone.hess_prod_updated
        update_hess_prod(cone)
    end
    Wi = cone.Wi
    @views mul!(prod[1:2, :], cone.hess[1:2, :], arr)
    @views mul!(prod[3:cone.dim, :], cone.hess[3:cone.dim, 1:2], arr[1:2, :])

    @inbounds for i in 1:size(arr, 2)
        vec_to_mat_U!(cone.mat, view(arr, 3:cone.dim, i))
        mul!(cone.mat2, Symmetric(cone.mat, :U), cone.Wi)
        mul!(cone.mat3, Symmetric(cone.Wi, :U), cone.mat2)
        @. cone.mat3 *=  cone.vzip1
        dot_prod = dot(Symmetric(cone.mat, :U), Symmetric(cone.Wivzi, :U)) # slow until dot product is merged in Julia
        @. cone.mat3 += cone.Wivzi * dot_prod
        mat_U_to_vec_scaled!(cone.vecn, cone.mat3)
        view(prod, 3:cone.dim, i) .+= cone.vecn
    end

    return prod
end
