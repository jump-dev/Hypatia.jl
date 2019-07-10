#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of sum of logarithms
(u in R, v in R_+, w in R_+^n) : u <= v*sum(log.(w/v))

barrier (guessed, reduces to 3-dim exp cone self-concordant barrier)
-log(v*sum(log.(w/v)) - u) - sum(log.(w)) - log(v)
=#

mutable struct HypoPerLog{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::AbstractVector{T}

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
    hess_fact # TODO prealloc

    function HypoPerLog{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

HypoPerLog{T}(dim::Int) where {T <: HypReal} = HypoPerLog{T}(dim, false)

# TODO maybe only allocate the fields we use
function setup_data(cone::HypoPerLog{T}) where {T <: HypReal}
    reset_data(cone)
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.tmp_hess = Symmetric(zeros(T, dim, dim), :U)
    cone.vwivlwvu = zeros(T, dim - 2)
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
    @. cone.vwivlwvu = v / cone.vlwvu / w
    cone.hess.data[1, 1] = abs2(cone.grad[1])
    lvwnivlwvu = (cone.lwv - d) / cone.vlwvu
    cone.hess.data[1, 2] = -lvwnivlwvu / cone.vlwvu
    @. cone.hess.data[1, 3:end] = -cone.vwivlwvu / cone.vlwvu
    cone.hess.data[2, 2] = abs2(lvwnivlwvu) + (d / cone.vlwvu + inv(v)) / v
    hden = (v * lvwnivlwvu - 1) / cone.vlwvu
    @. cone.hess.data[2, 3:end] = hden / w
    for j in 1:d
        j2 = 2 + j
        for i in 1:j
            cone.hess.data[2 + i, j2] = cone.vwivlwvu[i] * cone.vwivlwvu[j]
        end
        cone.hess.data[j2, j2] -= cone.grad[j2] / w[j]
    end
    cone.hess_updated = true
    return cone.hess
end
