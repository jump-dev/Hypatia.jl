#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of sum of logarithms
(u in R, v in R_+, w in R_+^n) : u <= v*sum(log.(w/v))

barrier (guessed, reduces to 3-dim exp cone self-concordant barrier)
-log(v*sum(log.(w/v)) - u) - sum(log.(w)) - log(v)
=#

mutable struct HypoPerLog{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
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

    lwv::T
    vlwvu::T
    vwivlwvu::Vector{T}
    tmp_hess::Symmetric{T, Matrix{T}}
    hess_fact
    hess_fact_cache

    function HypoPerLog{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 3
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

HypoPerLog{T}(dim::Int) where {T <: Real} = HypoPerLog{T}(dim, false)

# TODO maybe only allocate the fields we use
function setup_data(cone::HypoPerLog{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.vwivlwvu = zeros(T, dim - 2)
    cone.hess_fact_cache = nothing
    return
end

get_nu(cone::HypoPerLog) = cone.dim

function set_initial_point(arr::AbstractVector, cone::HypoPerLog)
    arr .= 1
    arr[1] = -1
    return arr
end

function update_feas(cone::HypoPerLog)
    @assert !cone.feas_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    if v > 0 && all(wi -> wi > 0, w)
        cone.lwv = sum(log(wi / v) for wi in w)
        cone.vlwvu = v * cone.lwv - u
        cone.is_feas = (cone.vlwvu > 0)
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::HypoPerLog)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    cone.grad[1] = inv(cone.vlwvu)
    cone.grad[2] = -(cone.lwv - (cone.dim - 2)) / cone.vlwvu - inv(v)
    gden = -1 - inv(cone.lwv - u / v)
    @. cone.grad[3:end] = gden / w
    cone.grad_updated = true
    return cone.grad
end

function update_hess(cone::HypoPerLog)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    d = cone.dim - 2
    H = cone.hess.data

    @. cone.vwivlwvu = v / cone.vlwvu / w
    H[1, 1] = abs2(cone.grad[1])
    lvwnivlwvu = (cone.lwv - d) / cone.vlwvu
    H[1, 2] = -lvwnivlwvu / cone.vlwvu
    @. H[1, 3:end] = -cone.vwivlwvu / cone.vlwvu
    H[2, 2] = abs2(lvwnivlwvu) + (d / cone.vlwvu + inv(v)) / v
    hden = (v * lvwnivlwvu - 1) / cone.vlwvu
    @. H[2, 3:end] = hden / w
    @inbounds for j in 1:d
        j2 = 2 + j
        @inbounds for i in 1:j
            H[2 + i, j2] = cone.vwivlwvu[i] * cone.vwivlwvu[j]
        end
        H[j2, j2] -= cone.grad[j2] / w[j]
    end

    cone.hess_updated = true
    return cone.hess
end
