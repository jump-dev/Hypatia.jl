#=
line search for s,z
=#

mutable struct LineSearcher{T <: Real}
    z::Vector{T}
    s::Vector{T}
    dual_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    primal_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
    skzk::Vector{T}
    nup1::T
    cone_times::Vector{Float64}
    cone_order::Vector{Int}

    function LineSearcher{T}(model::Models.Model{T}) where {T <: Real}
        cones = model.cones
        line_searcher = new{T}()
        z = line_searcher.z = zeros(T, model.q)
        s = line_searcher.s = zeros(T, model.q)
        line_searcher.dual_views = [view(Cones.use_dual_barrier(cone) ? s : z, idxs) for (cone, idxs) in zip(cones, model.cone_idxs)]
        line_searcher.primal_views = [view(Cones.use_dual_barrier(cone) ? z : s, idxs) for (cone, idxs) in zip(cones, model.cone_idxs)]
        line_searcher.skzk = zeros(T, length(cones))
        line_searcher.nup1 = T(model.nu + 1)
        line_searcher.cone_times = zeros(length(cones))
        line_searcher.cone_order = collect(1:length(cones))
        return line_searcher
    end
end

# backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
function find_max_alpha(
    point::Point{T},
    dir::Point{T},
    line_searcher::LineSearcher{T},
    model::Models.Model{T}; # TODO remove if not using
    prev_alpha::T,
    min_alpha::T,
    min_nbhd::T = T(0.01),
    # max_nbhd::T = one(T),
    max_nbhd::T = T(0.99),
    ) where {T <: Real}
    cones = model.cones
    cone_order = line_searcher.cone_order
    (tau, kap) = (point.tau[1], point.kap[1])
    (tau_dir, kap_dir) = (dir.tau[1], dir.kap[1])
    skzk = line_searcher.skzk

    alpha_reduce = T(0.95) # TODO tune, maybe try smaller for pred_alpha since heuristic

    # TODO experiment with starting alpha (<1)
    # alpha = one(T)
    alpha = max(T(0.1), min(prev_alpha * T(1.4), one(T))) # TODO option for parameter

    if tau_dir < zero(T)
        alpha = min(alpha, -tau / tau_dir)
    end
    if kap_dir < zero(T)
        alpha = min(alpha, -kap / kap_dir)
    end
    alpha *= T(0.9999)

    alpha /= alpha_reduce
    # TODO for feas, as soon as cone is feas, don't test feas again, since line search is backwards
    while true
        if alpha < min_alpha
            # alpha is very small so finish
            alpha = zero(T)
            break
        end
        alpha *= alpha_reduce

        taukap_ls = (tau + alpha * tau_dir) * (kap + alpha * kap_dir)
        (taukap_ls < eps(T)) && continue

        # order the cones by how long it takes to check neighborhood condition and iterate in that order, to improve efficiency
        sortperm!(cone_order, line_searcher.cone_times, initialized = true) # NOTE stochastic

        @. line_searcher.z = point.z + alpha * dir.z
        @. line_searcher.s = point.s + alpha * dir.s

        for k in cone_order
            skzk[k] = dot(line_searcher.primal_views[k], line_searcher.dual_views[k])
        end
        any(<(eps(T)), skzk) && continue

        mu_ls = (sum(skzk) + taukap_ls) / line_searcher.nup1
        (mu_ls < eps(T)) && continue

        min_nbhd_mu = min_nbhd * mu_ls
        (taukap_ls < min_nbhd_mu) && continue
        any(skzk[k] < min_nbhd_mu * Cones.get_nu(cones[k]) for k in cone_order) && continue

        # TODO experiment with SY nbhd for tau-kappa
        isfinite(max_nbhd) && (abs(taukap_ls - mu_ls) > max_nbhd * mu_ls) && continue

        rtmu = sqrt(mu_ls)
        irtmu = inv(rtmu)
        in_nbhd = true
        for k in cone_order
            cone_k = cones[k]
            time_k = time_ns()

            Cones.load_point(cone_k, line_searcher.primal_views[k], irtmu)
            Cones.load_dual_point(cone_k, line_searcher.dual_views[k])
            Cones.reset_data(cone_k)

            in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && Cones.in_neighborhood(cone_k, rtmu, max_nbhd))
            # in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && (isinf(max_nbhd) || Cones.in_neighborhood(cone_k, rtmu, max_nbhd)))
            # TODO is_dual_feas function should fall back to a nbhd-like check (for ray maybe) if not using nbhd check
            # in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k))

            line_searcher.cone_times[k] = time_ns() - time_k
            if !in_nbhd_k
                in_nbhd = false
                break
            end
        end
        in_nbhd && break
    end

    return alpha
end
