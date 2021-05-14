#=
primal dual point
=#

mutable struct Point{T <: Real}
    vec::Vector{T}

    x::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    y::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    z::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    tau::SubArray{T, 0, Vector{T}, Tuple{Int}, true}
    s::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    kap::SubArray{T, 0, Vector{T}, Tuple{Int}, true}

    ztsk::SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}
    z_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    s_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    dual_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    primal_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}

    Point{T}() where {T <: Real} = new{T}()
end

function Point(
    model::Models.Model{T};
    ztsk_only::Bool = false,
    ) where {T <: Real}
    point = Point{T}()
    (n, p, q) = (model.n, model.p, model.q)
    tau_idx = n + p + q + 1
    dim = tau_idx + q + 1
    vec = point.vec = zeros(T, tau_idx + q + 1)

    @views point.ztsk = vec[(n + p + 1):end]
    ztsk_only && return point

    point.tau = view(vec, tau_idx)
    point.kap = view(vec, dim)
    @views begin
        point.z = vec[n + p .+ (1:q)]
        point.s = vec[tau_idx .+ (1:q)]
        point.x = vec[1:n]
        point.y = vec[n .+ (1:p)]
    end

    point.z_views = [view(point.z, idxs) for idxs in model.cone_idxs]
    point.s_views = [view(point.s, idxs) for idxs in model.cone_idxs]
    point.dual_views = [Cones.use_dual_barrier(cone_k) ? point.s_views[k] :
        point.z_views[k] for (k, cone_k) in enumerate(model.cones)]
    point.primal_views = [Cones.use_dual_barrier(cone_k) ? point.z_views[k] :
        point.s_views[k] for (k, cone_k) in enumerate(model.cones)]

    return point
end
