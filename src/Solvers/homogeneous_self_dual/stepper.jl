#=
Copyright 2018, Chris Coey and contributors

interior point stepper and line search functions for algorithms based on homogeneous self dual embedding
=#

mutable struct CombinedHSDStepper{T <: Real} <: HSDStepper{T}
    system_solver::CombinedHSDSystemSolver{T}
    max_nbhd::T
    use_infty_nbhd::Bool

    prev_aff_alpha::T
    prev_gamma::T
    prev_alpha::T

    z_temp::Vector{T}
    s_temp::Vector{T}
    primal_views
    dual_views
    nbhd_temp
    cones_infeas::Vector{Bool}
    cones_loaded::Vector{Bool}

    function CombinedHSDStepper{T}(
        model::Models.LinearModel{T};
        system_solver::CombinedHSDSystemSolver{T} = (model isa Models.PreprocessedLinearModel{T} ? QRCholCombinedHSDSystemSolver{T}(model) : NaiveCombinedHSDSystemSolver{T}(model)),
        use_infty_nbhd::Bool = true,
        max_nbhd::T = T(0.7), # TODO tune: maybe (use_infty_nbhd ? T(0.5) : T(0.75))
        ) where {T <: Real}
        stepper = new{T}()

        stepper.system_solver = system_solver
        stepper.max_nbhd = max_nbhd
        stepper.use_infty_nbhd = use_infty_nbhd

        stepper.prev_aff_alpha = one(T)
        stepper.prev_gamma = one(T)
        stepper.prev_alpha = one(T)

        stepper.z_temp = similar(model.h)
        stepper.s_temp = similar(model.h)
        stepper.primal_views = [view(Cones.use_dual(model.cones[k]) ? stepper.z_temp : stepper.s_temp, model.cone_idxs[k]) for k in eachindex(model.cones)]
        stepper.dual_views = [view(Cones.use_dual(model.cones[k]) ? stepper.s_temp : stepper.z_temp, model.cone_idxs[k]) for k in eachindex(model.cones)]
        if !use_infty_nbhd
            stepper.nbhd_temp = [Vector{T}(undef, length(model.cone_idxs[k])) for k in eachindex(model.cones)]
        end
        stepper.cones_infeas = trues(length(model.cones))
        stepper.cones_loaded = trues(length(model.cones))

        return stepper
    end
end

function step(solver::HSDSolver{T}, stepper::CombinedHSDStepper{T}) where {T <: Real}
    model = solver.model
    point = solver.point

    # calculate affine/prediction and correction directions
    Cones.load_point.(solver.model.cones, solver.point.primal_views)
    @timeit solver.timer "directions" (x_pred, x_corr, y_pred, y_corr, z_pred, z_corr, s_pred, s_corr, tau_pred, tau_corr, kap_pred, kap_corr) = get_combined_directions(solver, stepper.system_solver)

    Cones.load_point.(solver.model.cones, stepper.primal_views)

    # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    # TODO try setting nbhd to T(Inf) and avoiding the neighborhood checks - requires tuning
    @timeit solver.timer "aff_alpha" aff_alpha = find_max_alpha_in_nbhd(z_pred, s_pred, tau_pred, kap_pred, stepper, solver, nbhd = one(T), prev_alpha = max(stepper.prev_aff_alpha, T(2e-2)), min_alpha = T(2e-2))
    stepper.prev_aff_alpha = aff_alpha

    gamma = abs2(one(T) - aff_alpha) # TODO allow different function (heuristic)
    stepper.prev_gamma = gamma

    # find distance alpha for stepping in combined direction
    z_comb = z_pred
    s_comb = s_pred
    pred_factor = one(T) - gamma
    @. z_comb = pred_factor * z_pred + gamma * z_corr
    @. s_comb = pred_factor * s_pred + gamma * s_corr
    tau_comb = pred_factor * tau_pred + gamma * tau_corr
    kap_comb = pred_factor * kap_pred + gamma * kap_corr
    @timeit solver.timer "comb_alpha" alpha = find_max_alpha_in_nbhd(z_comb, s_comb, tau_comb, kap_comb, stepper, solver, nbhd = stepper.max_nbhd, prev_alpha = stepper.prev_alpha, min_alpha = T(1e-2))

    if iszero(alpha)
        # could not step far in combined direction, so perform a pure correction step
        solver.verbose && println("performing correction step")
        z_comb = z_corr
        s_comb = s_corr
        tau_comb = tau_corr
        kap_comb = kap_corr
        @timeit solver.timer "corr_alpha" alpha = find_max_alpha_in_nbhd(z_comb, s_comb, tau_comb, kap_comb, stepper, solver, nbhd = stepper.max_nbhd, prev_alpha = one(T), min_alpha = T(1e-4))

        if iszero(alpha)
            error("could not step in correction direction; terminating")
        end
        @. point.x += alpha * x_corr
        @. point.y += alpha * y_corr
    else
        @. point.x += alpha * (pred_factor * x_pred + gamma * x_corr)
        @. point.y += alpha * (pred_factor * y_pred + gamma * y_corr)
    end
    stepper.prev_alpha = alpha

    # step distance alpha in combined direction
    @. point.z += alpha * z_comb
    @. point.s += alpha * s_comb
    solver.tau += alpha * tau_comb
    solver.kap += alpha * kap_comb
    calc_mu(solver)

    @assert solver.tau > zero(T) && solver.kap > zero(T) && solver.mu > zero(T)

    return point
end

function print_iteration_stats(solver::HSDSolver{T}, stepper::CombinedHSDStepper{T}) where {T <: Real}
    if iszero(solver.num_iters)
        @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
            "iter", "p_obj", "d_obj", "abs_gap", "rel_gap",
            "x_feas", "y_feas", "z_feas", "tau", "kap", "mu",
            "gamma", "alpha",
            )
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap, solver.rel_gap,
            solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu
            )
    else
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap, solver.rel_gap,
            solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu,
            stepper.prev_gamma, stepper.prev_alpha,
            )
    end
    flush(stdout)
end

# backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
function find_max_alpha_in_nbhd(
    z_dir::AbstractVector{T},
    s_dir::AbstractVector{T},
    tau_dir::T,
    kap_dir::T,
    stepper::CombinedHSDStepper{T},
    solver::HSDSolver{T};
    nbhd::T,
    prev_alpha::T,
    min_alpha::T,
    ) where {T <: Real}
    point = solver.point
    model = solver.model
    z_temp = stepper.z_temp
    s_temp = stepper.s_temp

    alpha = min(prev_alpha * T(1.4), one(T)) # TODO option for parameter
    if kap_dir < zero(T)
        alpha = min(alpha, -solver.kap / kap_dir)
    end
    if tau_dir < zero(T)
        alpha = min(alpha, -solver.tau / tau_dir)
    end
    alpha *= T(0.9999)

    stepper.cones_infeas .= true
    tau_temp = kap_temp = taukap_temp = mu_temp = zero(T)
    while true
        @timeit solver.timer "ls_update" begin
        @. z_temp = point.z + alpha * z_dir
        @. s_temp = point.s + alpha * s_dir
        tau_temp = solver.tau + alpha * tau_dir
        kap_temp = solver.kap + alpha * kap_dir
        taukap_temp = tau_temp * kap_temp
        mu_temp = (dot(s_temp, z_temp) + taukap_temp) / (one(T) + model.nu)
        end

        if mu_temp > zero(T)
            @timeit solver.timer "nbhd_check" in_nbhd = check_nbhd(mu_temp, taukap_temp, nbhd, model.cones, stepper)
            if in_nbhd
                break
            end
        end

        if alpha < min_alpha
            # alpha is very small so just let it be zero
            alpha = zero(T)
            break
        end

        # iterate is outside the neighborhood: decrease alpha
        alpha *= T(0.8) # TODO option for parameter
    end

    return alpha
end

function check_nbhd(
    mu_temp::T,
    taukap_temp::T,
    nbhd::T,
    cones::Vector{<:Cones.Cone{T}},
    stepper::CombinedHSDStepper{T},
    ) where {T <: Real}
    if isfinite(nbhd)
        rhs_nbhd = abs2(mu_temp * nbhd)
        lhs_nbhd = abs2(taukap_temp - mu_temp)
        if lhs_nbhd >= rhs_nbhd
            return false
        end
    end

    # accept primal iterate if it is inside the cone and neighborhood
    # first check inside cone for whichever cones were violated last line search iteration
    for (k, cone_k) in enumerate(cones)
        if stepper.cones_infeas[k]
            Cones.reset_data(cone_k)
            if Cones.is_feas(cone_k)
                stepper.cones_infeas[k] = false
                stepper.cones_loaded[k] = true
            else
                return false
            end
        else
            stepper.cones_loaded[k] = false
        end
    end

    if isfinite(nbhd)
        for (k, cone_k) in enumerate(cones)
            if !stepper.cones_loaded[k]
                Cones.reset_data(cone_k)
                if !Cones.is_feas(cone_k)
                    return false
                end
            end

            # modifies dual_views
            duals_k = stepper.dual_views[k]
            g_k = Cones.grad(cone_k)
            @. duals_k += mu_temp * g_k

            if stepper.use_infty_nbhd
                k_nbhd = abs2(norm(duals_k, Inf) / norm(g_k, Inf))
                # k_nbhd = abs2(maximum(abs(dj) / abs(gj) for (dj, gj) in zip(duals_k, g_k))) # TODO try this neighborhood
                lhs_nbhd = max(lhs_nbhd, k_nbhd)
            else
                nbhd_temp_k = stepper.nbhd_temp[k]
                Cones.inv_hess_prod!(nbhd_temp_k, duals_k, cone_k)
                k_nbhd = dot(duals_k, nbhd_temp_k)
                if k_nbhd <= -cbrt(eps(T))
                    println("numerical issue for cone: k_nbhd is $k_nbhd")
                    return false
                elseif k_nbhd > zero(T)
                    lhs_nbhd += k_nbhd
                end
            end

            if lhs_nbhd > rhs_nbhd
                return false
            end
        end
    end

    return true
end
