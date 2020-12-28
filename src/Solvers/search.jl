#=
step distance search helpers for s,z
=#

mutable struct StepSearcher{T <: Real}
    skzk::Vector{T}
    nup1::T
    cone_times::Vector{Float64}
    cone_order::Vector{Int}
    min_nbhd::T
    max_nbhd::T
    alpha_sched::Vector{T}

    function StepSearcher{T}(model::Models.Model{T}) where {T <: Real}
        cones = model.cones
        step_searcher = new{T}()
        step_searcher.skzk = zeros(T, length(cones))
        step_searcher.nup1 = T(model.nu + 1)
        step_searcher.cone_times = zeros(length(cones))
        step_searcher.cone_order = collect(1:length(cones))
        step_searcher.min_nbhd = T(0.01) # TODO tune
        step_searcher.max_nbhd = T(0.99) # TODO tune, maybe should be different for cones without third order correction
        step_searcher.alpha_sched = T[0.9999, 0.999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001, 0.0001] # TODO tune
        return step_searcher
    end
end

# backwards search on alphas in alpha schedule
function search_alpha(
    point::Point{T},
    model::Models.Model{T},
    stepper::Stepper{T};
    # prev_alpha::T = one(T), # TODO so don't try largest alpha first always
    min_alpha::T = zero(T),
    ) where {T <: Real}
    step_searcher = stepper.step_searcher
    for alpha in step_searcher.alpha_sched
        (alpha < min_alpha) && break # alpha is very small so finish
        update_cone_points(alpha, point, stepper) || continue
        check_cone_points(stepper.res, step_searcher, model) && return alpha
    end
    return zero(T)
end

function check_cone_points(
    cand::Point{T},
    step_searcher::StepSearcher{T},
    model::Models.Model{T},
    ) where {T <: Real}
    cone_order = step_searcher.cone_order
    skzk = step_searcher.skzk
    cones = model.cones
    max_nbhd = step_searcher.max_nbhd

    for k in cone_order
        skzk[k] = dot(cand.primal_views[k], cand.dual_views[k])
        (skzk[k] < eps(T)) && return false
    end
    taukap_cand = cand.tau[1] * cand.kap[1]
    mu_cand = (sum(skzk) + taukap_cand) / step_searcher.nup1
    (mu_cand < eps(T)) && return false

    min_nbhd_mu = step_searcher.min_nbhd * mu_cand
    (taukap_cand < min_nbhd_mu) && return false
    any(skzk[k] < min_nbhd_mu * Cones.get_nu(cones[k]) for k in cone_order) && return false
    isfinite(max_nbhd) && (abs(taukap_cand - mu_cand) > max_nbhd * mu_cand) && return false

    # order the cones by how long it takes to check neighborhood condition and iterate in that order, to improve efficiency
    sortperm!(cone_order, step_searcher.cone_times, initialized = true) # NOTE stochastic

    rtmu = sqrt(mu_cand)
    irtmu = inv(rtmu)
    for k in cone_order
        cone_k = cones[k]
        start_time = time()
        Cones.load_point(cone_k, cand.primal_views[k], irtmu)
        Cones.load_dual_point(cone_k, cand.dual_views[k])
        Cones.reset_data(cone_k)
        in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && (isinf(max_nbhd) || Cones.in_neighborhood(cone_k, rtmu, max_nbhd)))
        step_searcher.cone_times[k] = time() - start_time
        in_nbhd_k || return false
    end

    return true
end


#
# mutable struct LineSearcher{T <: Real}
#     z::Vector{T}
#     s::Vector{T}
#     dual_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
#     primal_views::Vector{SubArray{T, 1, Vector{T}, Tuple{UnitRange{Int}}, true}}
#     skzk::Vector{T}
#     nup1::T
#     cone_times::Vector{Float64}
#     cone_order::Vector{Int}
#
#     function LineSearcher{T}(model::Models.Model{T}) where {T <: Real}
#         cones = model.cones
#         line_searcher = new{T}()
#         z = line_searcher.z = zeros(T, model.q)
#         s = line_searcher.s = zeros(T, model.q)
#         line_searcher.dual_views = [view(Cones.use_dual_barrier(cone) ? s : z, idxs) for (cone, idxs) in zip(cones, model.cone_idxs)]
#         line_searcher.primal_views = [view(Cones.use_dual_barrier(cone) ? z : s, idxs) for (cone, idxs) in zip(cones, model.cone_idxs)]
#         line_searcher.skzk = zeros(T, length(cones))
#         line_searcher.nup1 = T(model.nu + 1)
#         line_searcher.cone_times = zeros(length(cones))
#         line_searcher.cone_order = collect(1:length(cones))
#         return line_searcher
#     end
# end
#
# # backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
# function find_max_alpha(
#     point::Point{T},
#     dir::Point{T},
#     line_searcher::LineSearcher{T},
#     model::Models.Model{T};
#     prev_alpha::T,
#     min_alpha::T,
#     min_nbhd::T = T(0.01),
#     max_nbhd::T = T(0.99),
#     ) where {T <: Real}
#     cones = model.cones
#     cone_order = line_searcher.cone_order
#     (tau, kap) = (point.tau[1], point.kap[1])
#     (tau_dir, kap_dir) = (dir.tau[1], dir.kap[1])
#     skzk = line_searcher.skzk
#
#     alpha_reduce = T(0.95) # TODO tune, maybe try smaller for pred_alpha since heuristic
#
#     # TODO experiment with starting alpha (<1)
#     alpha = max(T(0.1), min(prev_alpha * T(1.4), one(T))) # TODO option for parameter
#
#     if tau_dir < zero(T)
#         alpha = min(alpha, -tau / tau_dir)
#     end
#     if kap_dir < zero(T)
#         alpha = min(alpha, -kap / kap_dir)
#     end
#     alpha *= T(0.9999)
#
#     alpha /= alpha_reduce
#     # TODO for feas, as soon as cone is feas, don't test feas again, since line search is backwards
#     while true
#         if alpha < min_alpha
#             # alpha is very small so finish
#             alpha = zero(T)
#             break
#         end
#         alpha *= alpha_reduce
#
#         taukap_ls = (tau + alpha * tau_dir) * (kap + alpha * kap_dir)
#         (taukap_ls < eps(T)) && continue
#
#         # order the cones by how long it takes to check neighborhood condition and iterate in that order, to improve efficiency
#         sortperm!(cone_order, line_searcher.cone_times, initialized = true) # NOTE stochastic
#
#         @. line_searcher.z = point.z + alpha * dir.z
#         @. line_searcher.s = point.s + alpha * dir.s
#
#         for k in cone_order
#             skzk[k] = dot(line_searcher.primal_views[k], line_searcher.dual_views[k])
#         end
#         any(<(eps(T)), skzk) && continue
#
#         mu_ls = (sum(skzk) + taukap_ls) / line_searcher.nup1
#         (mu_ls < eps(T)) && continue
#
#         min_nbhd_mu = min_nbhd * mu_ls
#         (taukap_ls < min_nbhd_mu) && continue
#         any(skzk[k] < min_nbhd_mu * Cones.get_nu(cones[k]) for k in cone_order) && continue
#         isfinite(max_nbhd) && (abs(taukap_ls - mu_ls) > max_nbhd * mu_ls) && continue
#
#         rtmu = sqrt(mu_ls)
#         irtmu = inv(rtmu)
#         in_nbhd = true
#         for k in cone_order
#             cone_k = cones[k]
#             line_searcher.cone_times[k] = @elapsed begin
#                 Cones.load_point(cone_k, line_searcher.primal_views[k], irtmu)
#                 Cones.load_dual_point(cone_k, line_searcher.dual_views[k])
#                 Cones.reset_data(cone_k)
#                 in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && (isinf(max_nbhd) || Cones.in_neighborhood(cone_k, rtmu, max_nbhd)))
#             end
#             if !in_nbhd_k
#                 in_nbhd = false
#                 break
#             end
#         end
#         in_nbhd && break
#     end
#
#     return alpha
# end

# # curve search
# function search_alpha_curve(
#     point::Point{T},
#     dir_nocorr::Point{T},
#     dir_corr::Point{T},
#     step_searcher::StepSearcher{T},
#     model::Models.Model{T};
#     prev_alpha::T = one(T),
#     min_alpha::T = T(1e-4),
#     min_nbhd::T = T(0.01),
#     max_nbhd::T = T(0.99),
#     ) where {T <: Real}
#     cones = model.cones
#
#     skzk = zeros(T, length(cones))
#     nup1 = model.nu + 1
#
#     alpha_reduce = T(0.95)
#
#     alpha = T(0.9999)
#     alpha /= alpha_reduce
#     while true
#         if alpha < min_alpha
#             # alpha is very small so finish
#             alpha = zero(T)
#             break
#         end
#         alpha *= alpha_reduce
#
#         @. cand.vec = point.vec + alpha * (dir_nocorr + alpha * dir_corr)
#
#         min(cand.tau[1], cand.kap[1]) < eps(T) && continue
#         taukap_c = cand.tau[1] * cand.kap[1]
#         (taukap_c < eps(T)) && continue
#         for k in eachindex(cones)
#             skzk[k] = dot(cand.primal_views[k], cand.dual_views[k])
#         end
#         any(<(eps(T)), skzk) && continue
#
#         mu_c = (sum(skzk) + taukap_c) / nup1
#         (mu_c < eps(T)) && continue
#
#         min_nbhd_mu = min_nbhd * mu_c
#         (taukap_c < min_nbhd_mu) && continue
#         any(skzk[k] < min_nbhd_mu * Cones.get_nu(cone_k) for (k, cone_k) in enumerate(cones)) && continue
#         isfinite(max_nbhd) && (abs(taukap_c - mu_c) > max_nbhd * mu_c) && continue
#
#         rtmu = sqrt(mu_c)
#         irtmu = inv(rtmu)
#         in_nbhd = true
#         for (k, cone_k) in enumerate(cones)
#             Cones.load_point(cone_k, cand.primal_views[k], irtmu)
#             Cones.load_dual_point(cone_k, cand.dual_views[k])
#             Cones.reset_data(cone_k)
#             in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) && (isinf(max_nbhd) || Cones.in_neighborhood(cone_k, rtmu, max_nbhd)))
#             if !in_nbhd_k
#                 in_nbhd = false
#                 break
#             end
#         end
#         in_nbhd && break
#     end
#
#     return alpha
# end
