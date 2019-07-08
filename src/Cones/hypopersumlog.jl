#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

(closure of) hypograph of perspective of sum of logarithms
(u in R, v in R_+, w in R_+^n) : u <= v*sum(log.(w/v))

barrier (guessed, reduces to 3-dim exp cone self-concordant barrier)
-log(v*sum(log.(w/v)) - u) - sum(log.(w)) - log(v)

TODO
- rename to and replace 3D cone (hypoperlog)
- remove vw and other numerical optimization
=#

mutable struct HypoPerSumLog{T <: HypReal} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::AbstractVector{T}

    is_feas::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    lwv::T
    vlwvu::T
    F

    function HypoPerSumLog{T}(dim::Int, is_dual::Bool) where {T <: HypReal}
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

HypoPerSumLog{T}(dim::Int) where {T <: HypReal} = HypoPerSumLog{T}(dim, false)

# TODO maybe only allocate the fields we use
function setup_data(cone::HypoPerSumLog{T}) where {T <: HypReal}
    dim = cone.dim
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

get_nu(cone::HypoPerSumLog) = cone.dim

function set_initial_point(arr::AbstractVector, cone::HypoPerSumLog)
    arr .= 1
    arr[1] = -1
    return arr
end

reset_data(cone::HypoPerSumLog) = (cone.is_feas = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

function update_feas(cone::HypoPerSumLog)
    @assert !cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    if v > 0 && all(wi -> wi > 0, w)
        w = view(cone.point, 3:cone.dim)
        cone.lwv = sum(wi -> log(wi / v), w)
        cone.vlwvu = v * lwv - u
        cone.is_feas = (cone.vlwvu > 0)
    end
    return cone.is_feas
end

function update_grad(cone::HypoPerSumLog)
    @assert cone.is_feas
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)
    cone.lwvn = cone.lwv - T(cone.dim - 2)
    cone.grad[1] = inv(cone.vlwvu)
    cone.grad[2] = -cone.lwvn / cone.vlwvu - inv(v)
    gden = -1 - inv(cone.lwv - u / v)
    @. cone.grad[3:end] = gden / w
    cone.grad_updated = true
    return cone.grad
end

# TODO only work with upper triangle
function update_hess(cone::HypoPerSumLog)
    @assert cone.grad_updated
    u = cone.point[1]
    v = cone.point[2]
    w = view(cone.point, 3:cone.dim)

    @. cone.vwivlwvu = v / w / cone.vlwvu
    cone.hess[1, 1] = abs2(cone.grad[1])
    cone.hess[1, 2] = -cone.lwvn / cone.vlwvu / cone.vlwvu
    @. cone.hess[1, 3:end] = -cone.vwivlwvu / cone.vlwvu
    cone.hess[2, 2] = -cone.lwvn * cone.hess[1, 2] + ((cone.dim - 2) / cone.vlwvu + inv(v)) / v
    hden = (-v * cone.hess[1, 2] - 1) / cone.vlwvu
    @. cone.hess[2, 3:end] = hden / w
    for j in eachindex(cone.vonw)
        j2 = 2 + j
        for i in 1:j
            H[2 + i, j2] = cone.vwivlwvu[i] * cone.vwivlwvu[j]
        end
        H[j2, j2] -= cone.grad[j2] / w[j]
    end
    cone.hess_updated = true
    return cone.hess
end


# ivlwvu = inv(vlwvu)
# vw = v ./ w # TODO remove allocations
# ivlwvu2 = abs2(ivlwvu)

# # Hessian
# H[1, 1] = ivlwvu2
# H[1, 2] = -lwvn * ivlwvu2
# @. H[1, 3:end] = -vw * ivlwvu2
# H[2, 2] = abs2(lwvn) * ivlwvu2 + ivlwvu * (cone.dim - 2) / v + inv(abs2(v))
# @. H[2, 3:end] = vw * lwvn * ivlwvu2 - ivlwvu / w
# for j in 1:(cone.dim - 2)
#     for i in 1:(j - 1)
#         H[2 + i, 2 + j] = ivlwvu2 * vw[i] * vw[j]
#     end
#     H[2 + j, 2 + j] = abs2(vw[j]) * ivlwvu2 + vw[j] / w[j] * ivlwvu + inv(abs2(w[j]))
# end
