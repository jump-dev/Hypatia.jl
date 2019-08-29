#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

=#


mutable struct SOSConvexMono{T <: Real} <: Cone{T}
    use_dual::Bool
    dim::Int
    point::Vector{T}

    feas_updated::Bool
    grad_updated::Bool
    hess_updated::Bool
    inv_hess_updated::Bool
    is_feas::Bool
    grad::Vector{T}
    hess::Symmetric{T, Matrix{T}}
    inv_hess::Symmetric{T, Matrix{T}}

    dist::T

    function SOSConvexMono{T}(dim::Int, is_dual::Bool) where {T <: Real}
        @assert dim >= 2
        cone = new{T}()
        cone.use_dual = is_dual
        cone.dim = dim
        return cone
    end
end

SOSConvexMono{T}(dim::Int) where {T <: Real} = SOSConvexMono{T}(dim, false)

reset_data(cone::SOSConvexMono) = (cone.feas_updated = cone.grad_updated = cone.hess_updated = cone.inv_hess_updated = false)

# TODO maybe only allocate the fields we use
function setup_data(cone::SOSConvexMono{T}) where {T <: Real}
    reset_data(cone)
    dim = cone.dim
    cone.point = zeros(T, dim)
    cone.grad = zeros(T, dim)
    cone.hess = Symmetric(zeros(T, dim, dim), :U)
    cone.inv_hess = Symmetric(zeros(T, dim, dim), :U)
    return
end

get_nu(cone::SOSConvexMono) = 2

function set_initial_point(arr::AbstractVector, cone::SOSConvexMono)
    arr .= 0
    arr[1] = 1
    return arr
end

function update_feas(cone::SOSConvexMono)
    @assert !cone.feas_updated
    u = cone.point[1]
    if u > 0
        w = view(cone.point, 2:cone.dim)
        cone.dist = (abs2(u) - sum(abs2, w)) / 2
        cone.is_feas = (cone.dist > 0)
    else
        cone.is_feas = false
    end
    cone.feas_updated = true
    return cone.is_feas
end

function update_grad(cone::SOSConvexMono)
    @assert cone.is_feas
    @. cone.grad = cone.point / cone.dist
    cone.grad[1] *= -1
    cone.grad_updated = true
    return cone.grad
end

# TODO only work with upper triangle
function update_hess(cone::SOSConvexMono)
    @assert cone.grad_updated
    mul!(cone.hess.data, cone.grad, cone.grad')
    cone.hess += inv(cone.dist) * I
    cone.hess[1, 1] -= 2 / cone.dist
    cone.hess_updated = true
    return cone.hess
end

# TODO only work with upper triangle
function update_inv_hess(cone::SOSConvexMono)
    @assert cone.is_feas
    mul!(cone.inv_hess.data, cone.point, cone.point')
    cone.inv_hess += cone.dist * I
    cone.inv_hess[1, 1] -= 2 * cone.dist
    cone.inv_hess_updated = true
    return cone.inv_hess
end
