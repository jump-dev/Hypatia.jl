#=
step distance search helpers for s,z
=#

mutable struct StepSearcher{T <: Real}
    min_prox::T
    prox_bound::T
    use_sum_prox::Bool
    alpha_sched::Vector{T}

    skzk::Vector{T}
    nup1::T
    cone_times::Vector{Float64}
    cone_order::Vector{Int}
    prev_sched::Int

    function StepSearcher{T}(
        model::Models.Model{T};
        min_prox::T = T(0.01),
        prox_bound::T = T(0.99),
        use_sum_prox::Bool = false,
        alpha_sched::Vector{T} = default_alpha_sched(T),
        ) where {T <: Real}
        cones = model.cones
        searcher = new{T}()
        searcher.min_prox = min_prox
        searcher.prox_bound = prox_bound
        searcher.use_sum_prox = use_sum_prox
        searcher.alpha_sched = alpha_sched
        searcher.skzk = zeros(T, length(cones))
        searcher.nup1 = T(model.nu + 1)
        searcher.cone_times = zeros(length(cones))
        searcher.cone_order = collect(1:length(cones))
        searcher.prev_sched = 0
        return searcher
    end
end

default_alpha_sched(T::Type{<:Real}) = T[
    0.9999, 0.999, 0.99, 0.97, 0.95, 0.9, 0.85, 0.8, 0.7, 0.6, 0.5,
    0.3, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0005]

# backwards search on alphas in alpha schedule
function search_alpha(
    point::Point{T},
    model::Models.Model{T},
    stepper::Stepper{T};
    sched::Int = start_sched(stepper, stepper.searcher)
    ) where {T <: Real}
    searcher = stepper.searcher

    while sched <= length(searcher.alpha_sched)
        alpha = searcher.alpha_sched[sched]
        # update ztsk only in stepper.temp
        update_stepper_points(alpha, point, stepper, true)

        # NOTE updates cone points and grad
        if check_cone_points(model, stepper)
            searcher.prev_sched = sched
            return alpha
        end
        sched += 1
    end

    searcher.prev_sched = sched
    return zero(T)
end

# fallback starts at first alpha in schedule
start_sched(stepper::Stepper, searcher::StepSearcher) = 1

function check_cone_points(
    model::Models.Model{T},
    stepper::Stepper{T};
    ) where {T <: Real}
    searcher = stepper.searcher
    cand = stepper.temp
    skzk = searcher.skzk
    cones = model.cones
    prox_bound = searcher.prox_bound
    min_prox = searcher.min_prox
    use_sum_prox = searcher.use_sum_prox

    taukap_cand = cand.tau[] * cand.kap[]
    (min(cand.tau[], cand.kap[], taukap_cand) < eps(T)) && return false

    for k in eachindex(cones)
        skzk[k] = dot(cand.primal_views[k], cand.dual_views[k])
        (skzk[k] < eps(T)) && return false
    end
    mu_cand = (sum(skzk) + taukap_cand) / searcher.nup1

    taukap_rel = taukap_cand / mu_cand
    taukap_prox = abs(taukap_rel - 1)
    if (mu_cand < eps(T)) || (taukap_rel < min_prox) || (taukap_prox > prox_bound)
        return false
    end
    prox_bound_sqr = abs2(prox_bound)
    for k in eachindex(cones)
        nu_k = Cones.get_nu(cones[k])
        skzkmu = skzk[k] / mu_cand
        if (skzkmu < min_prox * nu_k) || (abs2(skzkmu - nu_k) > prox_bound_sqr * nu_k)
            return false
        end
    end

    # order the cones by how long it takes to check proximity condition and
    # iterate in that order, to improve efficiency
    cone_order = searcher.cone_order
    sortperm!(cone_order, searcher.cone_times, initialized = true) # stochastic

    sum_prox = taukap_prox
    rtmu = sqrt(mu_cand)
    irtmu = inv(rtmu)
    for k in cone_order
        cone_k = cones[k]
        start_time = time()
        Cones.load_point(cone_k, cand.primal_views[k], irtmu)
        Cones.load_dual_point(cone_k, cand.dual_views[k])
        Cones.reset_data(cone_k)

        in_prox_k = false
        if Cones.is_feas(cone_k) && Cones.is_dual_feas(cone_k) &&
            Cones.check_numerics(cone_k)
            prox_k = Cones.get_proximity(cone_k, rtmu, use_sum_prox)
            if !isnan(prox_k) && (prox_k < prox_bound)
                sum_prox += prox_k
                if !use_sum_prox || (sum_prox < prox_bound)
                    in_prox_k = true
                end
            end
        end
        searcher.cone_times[k] = time() - start_time
        in_prox_k || return false
    end

    return true
end
