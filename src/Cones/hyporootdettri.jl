#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

hypograph of root determinant
(u in R, W in S_n+) : u <= det(W)^(1/n)

barrier from correspondence with A. Nemirovski
-(5 / 3) ^ 2 * (log(det(W) ^ (1 / n) - u) + logdet(W))

TODO needs updating to smat/svec as on dev branch
TODO hess_prod.
this is very similar to logdet situation. part of the hessian has a kronecker form.
=#

mutable struct HypoRootDetTri{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    side::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    inv_hess_prod_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}
    hess_fact_cache

    W::Matrix{T}
    work_mat::Matrix{T}
    fact_W
    Wi::Matrix{T}
    rootdet::T
    rootdetu::T
    frac::T
    twentyfive_ninths::T

    function HypoRootDetTri{T}(
        dim::Int,
        is_dual::Bool;
        hess_fact_cache = hessian_cache(T),
        ) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        cone.hess_fact_cache = hess_fact_cache
        return cone
    end
end

HypoRootDetTri{T}(dim::Int) where {T <: Real} = HypoRootDetTri{T}(dim, false)

function setup_data(cone::HypoRootDetTri{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.side = round(Int, sqrt(0.25 + 2 * (dim - 1)) - 0.5)
    cone.twentyfive_ninths = T(25) / T(9)
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    load_matrix(cone.hess_fact_cache, cone.hess)
    cone.W = zeros(T, cone.side, cone.side)
    cone.work_mat = zeros(T, cone.side, cone.side)
    return
end

get_nu(cone::HypoRootDetTri) = (cone.side + 1) * cone.twentyfive_ninths

# TODO not everything needs a T(.), what's the style?
function set_initial_point(arr::AbstractVector, cone::HypoRootDetTri{T}) where {T}
    arr .= 0
    n = cone.side
    fact1 = sqrt(T(5) * abs2(T(n)) + T(2) * T(n) + one(T))
    fact2 = sqrt((T(3) * T(n) - fact1 + one(T)) / (T(n) + one(T)))
    arr[1] = -T(5) * fact2 / T(3) / sqrt(T(2))
    k = 2
    @inbounds for i in 1:cone.side
        arr[k] = T(5) * fact2 * (T(n) + fact1 + one(T)) / T(n) / T(6) / sqrt(T(2))
        k += i + 1
    end
    return arr
end

function update_feas(cone::HypoRootDetTri)
    @assert !cone.feas_updated
    u = cone.point[1]

    vec_to_mat_U!(cone.W, view(cone.point, 2:cone.dim))
    copyto!(cone.work_mat, cone.W)
    cone.fact_W = cholesky!(Symmetric(cone.work_mat, :U), check = false)
    if isposdef(cone.fact_W)
        cone.rootdet = det(cone.fact_W) ^ (inv(cone.side))
        cone.rootdetu = cone.rootdet - u
        cone.is_feas = cone.rootdetu > 0
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoRootDetTri)
    @assert cone.is_feas
    u = cone.point[1]
    W = cone.W

    cone.grad[1] = inv(cone.rootdetu)
    cone.Wi = inv(cone.fact_W)
    @views mat_U_to_vec_scaled!(cone.grad[2:cone.dim], cone.Wi)
    cone.frac = cone.rootdet / cone.side / cone.rootdetu
    @. @views cone.grad[2:cone.dim] *= -(cone.frac + 1)
    @. cone.grad *= cone.twentyfive_ninths
    cone.grad_updated = true
    return cone.grad
end

# TODO refactor and save fields from gradient
function update_hess(cone::HypoRootDetTri)
    @assert cone.grad_updated
    u = cone.point[1]
    rootdetu = cone.rootdetu
    # rootdet / rootdetu / side
    frac = cone.frac
    Wi = cone.Wi
    hess = cone.hess.data

    hess[1, 1] = cone.grad[1] / rootdetu
    @views mat_U_to_vec_scaled!(hess[1, 2:cone.dim], Wi)
    # will scale by 25 / 9 at the end
    @. hess[1, 2:end] *= -frac / rootdetu

    const1 = frac + 1
    const2 = (abs2(frac) - frac / cone.side)
    k1 = 2
    for i in 1:cone.side, j in 1:i
        k2 = 2
        @inbounds for i2 in 1:cone.side, j2 in 1:i2
            if (i == j) && (i2 == j2)
                hess[k2, k1] = abs2(Wi[i2, i]) * const1 + Wi[i, i] * Wi[i2, i2] * const2
            elseif (i != j) && (i2 != j2)
                hess[k2, k1] = 2 * (Wi[i2, i] * Wi[j, j2] + Wi[j2, i] * Wi[j, i2]) * const1 + 4 * Wi[i, j] * Wi[i2, j2] * const2
            else
                hess[k2, k1] = 2 * (Wi[i2, i] * Wi[j, j2] * const1 + Wi[i, j] * Wi[i2, j2] * const2)
            end
            if k2 == k1
                break
            end
            k2 += 1
        end
        k1 += 1
    end
    @. @views hess[1:end, 2:end] *= cone.twentyfive_ninths

    cone.hess_updated = true
    return cone.hess
end
