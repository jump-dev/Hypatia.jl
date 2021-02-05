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
        step_searcher.alpha_sched = T[0.9999, 0.999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5, 0.3, 0.1] # TODO tune
        return step_searcher
    end
end

# backwards search on alphas in alpha schedule
function search_alpha(
    point::Point{T},
    model::Models.Model{T},
    stepper::Stepper{T};
    ) where {T <: Real}
    step_searcher = stepper.step_searcher
    for alpha in step_searcher.alpha_sched
        update_cone_points(alpha, point, stepper, true) # update ztsk only
        check_cone_points(stepper.temp, step_searcher, model) && return alpha
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

    taukap_cand = cand.tau[] * cand.kap[]
    (min(cand.tau[], cand.kap[], taukap_cand) < eps(T)) && return false

    for k in cone_order
        skzk[k] = dot(cand.primal_views[k], cand.dual_views[k])
        (skzk[k] < eps(T)) && return false
    end
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
