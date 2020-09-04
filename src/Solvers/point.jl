#=
primal dual point
=#

mutable struct Point{T <: Real}
    vec::Vector{T}

    x#::Vector{T}
    y#::Vector{T}
    z#::Vector{T}
    tau#::T
    s#::Vector{T}
    kap#::T

    z_views::Vector#{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    s_views::Vector#{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    dual_views::Vector#{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    primal_views::Vector#{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}

    function Point(model::Models.Model{T}) where {T <: Real}
        point = new{T}()
        (n, p, q) = (model.n, model.p, model.q)

        tau_idx = n + p + q + 1
        vec = point.vec = zeros(T, tau_idx + q + 1)
        @views begin
            point.x = vec[1:n]
            point.y = vec[n .+ (1:p)]
            point.z = vec[n + p .+ (1:q)]
            point.tau = vec[tau_idx:tau_idx]
            point.s = vec[tau_idx .+ (1:q)]
            point.kap = vec[end:end]
        end

        point.z_views = [view(point.z, idxs) for idxs in model.cone_idxs]
        point.s_views = [view(point.s, idxs) for idxs in model.cone_idxs]
        point.dual_views = [Cones.use_dual_barrier(cone_k) ? point.s_views[k] : point.z_views[k] for (k, cone_k) in enumerate(model.cones)]
        point.primal_views = [Cones.use_dual_barrier(cone_k) ? point.z_views[k] : point.s_views[k] for (k, cone_k) in enumerate(model.cones)]

        return point
    end

    # function Point(
    #     x::Vector{T},
    #     y::Vector{T},
    #     z::Vector{T},
    #     s::Vector{T},
    #     cones::Vector{<:Cones.Cone{T}},
    #     cone_idxs::Vector{UnitRange{Int}};
    #     tau::T = one(T),
    #     kap::T = one(T),
    #     ) where {T <: Real}
    #     point = new{T}()
    #
    #     point.x = x
    #     point.y = y
    #     point.z = z
    #     point.s = s
    #     point.tau = tau
    #     point.kap = kap
    #
    #     point.z_views = [view(point.z, idxs) for idxs in cone_idxs]
    #     point.s_views = [view(point.s, idxs) for idxs in cone_idxs]
    #     point.dual_views = [Cones.use_dual_barrier(cones[k]) ? point.s_views[k] : point.z_views[k] for k in eachindex(cones)]
    #     point.primal_views = [Cones.use_dual_barrier(cones[k]) ? point.z_views[k] : point.s_views[k] for k in eachindex(cones)]
    #
    #     return point
    # end
end
