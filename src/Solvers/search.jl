#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
step distance search helpers for s,z
=#

mutable struct StepSearcher{T <: Real}
    min_prox::T
    prox_bound::T
    use_max_prox::Bool
    alpha_sched::Vector{T}

    szk::Vector{T}
    nup1::T
    cone_times::Vector{Float64}
    cone_order::Vector{Int}
    prev_sched::Int
    prox::T

    function StepSearcher{T}(
        model::Models.Model{T};
        min_prox::T = T(0.01),
        prox_bound::T = T(0.99),
        use_max_prox::Bool = true,
        alpha_sched::Vector{T} = default_alpha_sched(T),
    ) where {T <: Real}
        cones = model.cones
        searcher = new{T}()
        searcher.min_prox = min_prox
        searcher.prox_bound = prox_bound
        searcher.use_max_prox = use_max_prox
        searcher.alpha_sched = alpha_sched
        searcher.szk = zeros(T, length(cones))
        searcher.nup1 = T(model.nu + 1)
        searcher.cone_times = zeros(length(cones))
        searcher.cone_order = collect(1:length(cones))
        searcher.prev_sched = 0
        searcher.prox = zero(T)
        return searcher
    end
end

function default_alpha_sched(T::Type{<:Real})
    return T[
        0.9999,
        0.999,
        0.99,
        0.97,
        0.95,
        0.9,
        0.85,
        0.8,
        0.7,
        0.6,
        0.5,
        0.3,
        0.1,
        0.05,
        0.01,
        0.005,
        0.001,
        0.0005,
    ]
end

# backwards search on alphas in alpha schedule
function search_alpha(
    point::Point{T},
    model::Models.Model{T},
    stepper::Stepper{T};
    sched::Int = start_sched(stepper, stepper.searcher),
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

function check_cone_points(model::Models.Model{T}, stepper::Stepper{T};) where {T <: Real}
    searcher = stepper.searcher
    cand = stepper.temp
    szk = searcher.szk
    cones = model.cones
    min_prox = searcher.min_prox
    use_max_prox = searcher.use_max_prox
    proxsqr_bound = abs2(searcher.prox_bound)

    taukap = cand.tau[] * cand.kap[]
    (min(cand.tau[], cand.kap[], taukap) < eps(T)) && return false

    for k in eachindex(cones)
        szk[k] = dot(cand.primal_views[k], cand.dual_views[k])
        (szk[k] < eps(T)) && return false
    end
    mu = (sum(szk) + taukap) / searcher.nup1
    (mu < eps(T)) && return false

    taukap_rel = taukap / mu
    (taukap_rel < min_prox) && return false
    taukap_proxsqr = abs2(taukap_rel - 1)
    (taukap_proxsqr > proxsqr_bound) && return false

    for k in eachindex(cones)
        nu_k = Cones.get_nu(cones[k])
        sz_rel_k = szk[k] / (mu * nu_k)
        if (sz_rel_k < min_prox) || (nu_k * abs2(sz_rel_k - 1) > proxsqr_bound)
            return false
        end
    end

    # order the cones by how long it takes to check proximity condition and
    # iterate in that order, to improve efficiency
    cone_order = searcher.cone_order
    sortperm!(cone_order, searcher.cone_times, initialized = true) # stochastic

    irtmu = inv(sqrt(mu))
    agg_proxsqr = taukap_proxsqr
    aggfun = (use_max_prox ? max : +)

    for k in cone_order
        cone_k = cones[k]
        start_time = time()
        Cones.load_point(cone_k, cand.primal_views[k], irtmu)
        Cones.load_dual_point(cone_k, cand.dual_views[k])
        Cones.reset_data(cone_k)

        in_prox_k = false
        if Cones.is_feas(cone_k) &&
           Cones.is_dual_feas(cone_k) &&
           Cones.check_numerics(cone_k)
            proxsqr_k = Cones.get_proxsqr(cone_k, irtmu, use_max_prox)
            agg_proxsqr = aggfun(agg_proxsqr, proxsqr_k)
            in_prox_k = (agg_proxsqr < proxsqr_bound)
        end
        searcher.cone_times[k] = time() - start_time
        in_prox_k || return false
    end

    searcher.prox = sqrt(agg_proxsqr)
    return true
end
