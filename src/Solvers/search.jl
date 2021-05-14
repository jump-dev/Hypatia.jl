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
    prev_sched::Int

    function StepSearcher{T}(model::Models.Model{T}) where {T <: Real}
        cones = model.cones
        step_searcher = new{T}()
        step_searcher.skzk = zeros(T, length(cones))
        step_searcher.nup1 = T(model.nu + 1)
        step_searcher.cone_times = zeros(length(cones))
        step_searcher.cone_order = collect(1:length(cones))
        step_searcher.min_nbhd = T(0.01) # TODO tune
        step_searcher.max_nbhd = T(0.99) # TODO tune, maybe should be different for cones without third order correction
        step_searcher.alpha_sched = T[ # TODO tune
            0.9999, 0.999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5,
            0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]
        step_searcher.prev_sched = 0
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
    sched = start_sched(stepper, step_searcher)
    while sched <= length(step_searcher.alpha_sched)
        alpha = step_searcher.alpha_sched[sched]
        update_cone_points(alpha, point, stepper, true) # update ztsk only
        if check_cone_points(stepper.temp, step_searcher, model)
            step_searcher.prev_sched = sched
            return alpha
        end
        sched += 1
    end
    step_searcher.prev_sched = sched
    return zero(T)
end

start_sched(stepper::Stepper, step_searcher::StepSearcher) = 1 # fallback starts at first alpha in schedule

function check_cone_points(
    cand::Point{T},
    step_searcher::StepSearcher{T},
    model::Models.Model{T},
    ) where {T <: Real}
    cone_order = step_searcher.cone_order
    skzk = step_searcher.skzk
    cones = model.cones
    max_nbhd = step_searcher.max_nbhd
    min_nbhd = step_searcher.min_nbhd

    taukap_cand = cand.tau[] * cand.kap[]
    (min(cand.tau[], cand.kap[], taukap_cand) < eps(T)) && return false

    for k in cone_order
        skzk[k] = dot(cand.primal_views[k], cand.dual_views[k])
        (skzk[k] < eps(T)) && return false
    end
    mu_cand = (sum(skzk) + taukap_cand) / step_searcher.nup1

    if (mu_cand < eps(T)) || (taukap_cand < min_nbhd * mu_cand) ||
        (abs(taukap_cand - mu_cand) > max_nbhd * mu_cand)
        return false
    end
    max_nbhd_sqr = abs2(max_nbhd)
    for k in cone_order
        nu_k = Cones.get_nu(cones[k])
        skzkmu = skzk[k] / mu_cand
        if (skzkmu < min_nbhd * nu_k) || (abs2(skzkmu - nu_k) > max_nbhd_sqr * nu_k)
            return false
        end
    end

    # order the cones by how long it takes to check neighborhood condition and iterate in that order, to improve efficiency
    sortperm!(cone_order, step_searcher.cone_times, initialized = true) # stochastic

    rtmu = sqrt(mu_cand)
    irtmu = inv(rtmu)
    for k in cone_order
        cone_k = cones[k]
        start_time = time()
        Cones.load_point(cone_k, cand.primal_views[k], irtmu)
        Cones.load_dual_point(cone_k, cand.dual_views[k])
        Cones.reset_data(cone_k)
        in_nbhd_k = (Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) &&
            Cones.in_neighborhood(cone_k, rtmu, max_nbhd))
        step_searcher.cone_times[k] = time() - start_time
        in_nbhd_k || return false
    end

    return true
end
