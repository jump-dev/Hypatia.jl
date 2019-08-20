#=
Copyright 2018, Chris Coey and contributors

hypograph of generalized geomean (product of powers) parametrized by alpha in R_+^n on unit simplex
(u in R, w in R_+^n) : u <= prod_i(w_i^alpha_i)
where sum_i(alpha_i) = 1, alpha_i >= 0

dual barrier (modified by reflecting around u = 0 and using dual cone definition) from "On self-concordant barriers for generalized power cones" by Roy & Xiao 2018
-log(prod_i((w_i/alpha_i)^alpha_i) + u) - sum_i((1 - alpha_i)*log(w_i/alpha_i)) - log(-u)

TODO try to make barrier evaluation more efficient
=#

mutable struct HypoGeomean{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    alpha::Vector{T}
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

    wiaa::T
    wiw::T
    alphaiw::Vector{T}
    a1ww::Vector{T}
    tmpnn::Matrix{T}
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact # TODO prealloc

    function HypoGeomean{T}(alpha::Vector{T}, is_dual::Bool) where {T <: Real}
        dim = length(alpha) + 1
        @assert dim >= 3
        @assert all(ai > 0 for ai in alpha)
        tol = 1e3 * eps(T)
        @assert sum(alpha) ≈ 1 atol=tol rtol=tol
        cone = new{T}()
        cone.use_dual = !is_dual # using dual barrier
        cone.dim = dim
        cone.alpha = alpha
        return cone
    end
end

HypoGeomean{T}(alpha::Vector{T}) where {T <: Real} = HypoGeomean{T}(alpha, false)

function setup_data(cone::HypoGeomean{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.a1ww = zeros(T, dim - 1)
    cone.alphaiw = zeros(T, dim - 1)
    cone.tmpnn = zeros(T, dim - 1, dim - 1)
    return
end

get_nu(cone::HypoGeomean) = cone.dim

function set_initial_point(arr::AbstractVector, cone::HypoGeomean)
    arr .= 1
    arr[1] = -prod(cone.alpha[i]^(-cone.alpha[i]) for i in eachindex(cone.alpha)) / cone.dim
    return arr
end

function update_feas(cone::HypoGeomean)
    @assert !cone.feas_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    @. cone.alphaiw = cone.alpha / w
    if u < 0 && all(wi -> wi > 0, w)
        cone.wiaa = -sum(cone.alpha[i] * log(cone.alphaiw[i]) for i in eachindex(cone.alpha))
        cone.is_feas = (cone.wiaa > log(-u))
        cone.wiaa = exp(cone.wiaa)
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoGeomean)
    @assert cone.is_feas
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    wiaau = cone.wiaa + u
    cone.wiw = cone.wiaa / wiaau
    @. cone.a1ww = cone.alphaiw * (1 - cone.wiw)
    cone.grad[1] = -inv(wiaau) - inv(u)
    @. cone.grad[2:end] = cone.a1ww - inv(w)
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoGeomean)
    @assert cone.grad_updated
    u = cone.point[1]
    w = view(cone.point, 2:cone.dim)
    H = cone.hess.data

    wiaau = cone.wiaa + u
    H[1, 1] = inv(wiaau) / wiaau + inv(u) / u
    @inbounds for j in eachindex(w)
        j1 = j + 1
        wiwaw = -cone.wiw * cone.alpha[j] / w[j]
        H[1, j1] = -wiwaw / wiaau
        @inbounds for i in 1:(j - 1)
            H[i + 1, j1] = wiwaw * cone.a1ww[i]
        end
        H[j1, j1] = wiwaw * cone.grad[j1] + (1 - cone.alpha[j]) / w[j] / w[j]
    end

    cone.hess_updated = true
    return cone.hess
end
