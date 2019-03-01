
mutable struct CombinedHSDStepper <: HSDStepper
    system_solver::CombinedHSDSystemSolver
    max_nbhd::Float64
    prev_alpha::Float64
    prev_gamma::Float64
    prev_affine_alpha::Float64
    z_temp::Vector{Float64}
    s_temp::Vector{Float64}
    primal_views
    dual_views
    nbhd_temp
    cones_outside_nbhd::Vector{Bool}
    cones_loaded::Vector{Bool}

    function CombinedHSDStepper(
        model::Models.LinearModel;
        system_solver::CombinedHSDSystemSolver = (model isa Models.PreprocessedLinearModel ? QRCholCombinedHSDSystemSolver(model) : NaiveCombinedHSDSystemSolver(model)),
        max_nbhd::Float64 = 0.75,
        )
        stepper = new()

        stepper.system_solver = system_solver
        stepper.max_nbhd = max_nbhd
        stepper.prev_alpha = 0.9999
        stepper.prev_gamma = 0.9999
        stepper.prev_affine_alpha = 0.9999

        stepper.z_temp = similar(model.h)
        stepper.s_temp = similar(model.h)
        stepper.primal_views = [view(Cones.use_dual(model.cones[k]) ? stepper.z_temp : stepper.s_temp, model.cone_idxs[k]) for k in eachindex(model.cones)]
        stepper.dual_views = [view(Cones.use_dual(model.cones[k]) ? stepper.s_temp : stepper.z_temp, model.cone_idxs[k]) for k in eachindex(model.cones)]
        stepper.nbhd_temp = [Vector{Float64}(undef, length(model.cone_idxs[k])) for k in eachindex(model.cones)]

        stepper.cones_outside_nbhd = trues(length(model.cones))
        stepper.cones_loaded = trues(length(model.cones))

        return stepper
    end
end

function combined_predict_correct(solver::HSDSolver, stepper::CombinedHSDStepper)
    model = solver.model
    point = solver.point

    # calculate affine/prediction and correction directions
    (x_dirs, y_dirs, z_dirs, s_dirs, tau_dirs, kap_dirs) = get_combined_directions(solver, stepper.system_solver)

    # calculate correction factor gamma by finding distance affine_alpha for stepping in affine direction
    affine_alpha = find_max_alpha_in_nbhd(z_dirs[:, 1], s_dirs[:, 1], tau_dirs[1], kap_dirs[1], 0.9999, stepper.prev_affine_alpha, stepper, solver)
    stepper.prev_affine_alpha = affine_alpha
    gamma = (1.0 - affine_alpha)^3 # TODO allow different function (heuristic)
    stepper.prev_gamma = gamma

    # find distance alpha for stepping in combined direction
    comb_scaling = [1.0 - gamma, gamma]
    z_comb = z_dirs * comb_scaling
    s_comb = s_dirs * comb_scaling
    tau_comb = dot(tau_dirs, comb_scaling)
    kap_comb = dot(kap_dirs, comb_scaling)
    alpha = find_max_alpha_in_nbhd(z_comb, s_comb, tau_comb, kap_comb, stepper.max_nbhd, stepper.prev_alpha, stepper, solver)

    if iszero(alpha)
        # could not step far in combined direction, so perform a pure correction step
        alpha = 0.99 # TODO assumes this maintains feasibility
        comb_scaling = [0.0, 1.0]
        z_comb = z_dirs * comb_scaling
        s_comb = s_dirs * comb_scaling
        tau_comb = dot(tau_dirs, comb_scaling)
        kap_comb = dot(kap_dirs, comb_scaling)
    end
    stepper.prev_alpha = alpha

    # step distance alpha in combined direction
    x_comb = x_dirs * comb_scaling
    y_comb = y_dirs * comb_scaling
    @. point.x += alpha * x_comb
    @. point.y += alpha * y_comb
    @. point.z += alpha * z_comb
    @. point.s += alpha * s_comb
    solver.tau += alpha * tau_comb
    solver.kap += alpha * kap_comb
    calc_mu(solver)

    @assert solver.tau > 0.0 && solver.kap > 0.0 && solver.mu > 0.0

    return point
end

function print_iteration_stats(solver::HSDSolver, stepper::CombinedHSDStepper)
    if solver.num_iters == 0
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
# TODO try infinite norm neighborhood, which is cheaper to check, or enforce that for each cone we are within a smaller neighborhood separately
function find_max_alpha_in_nbhd(z_dir::AbstractVector{Float64}, s_dir::AbstractVector{Float64}, tau_dir::Float64, kap_dir::Float64, nbhd::Float64, prev_alpha::Float64, stepper::CombinedHSDStepper, solver::HSDSolver)
    point = solver.point
    model = solver.model
    cones = model.cones
    z_temp = stepper.z_temp
    s_temp = stepper.s_temp

    # alpha = 0.9999 # TODO make this an option
    alpha = min(prev_alpha * 1.4, 0.9999)

    if kap_dir < 0.0
        alpha = min(alpha, -solver.kap / kap_dir)
    end
    if tau_dir < 0.0
        alpha = min(alpha, -solver.tau / tau_dir)
    end
    # TODO what about mu? quadratic equation. need dot(s_temp, z_temp) + tau_temp * kap_temp > 0

    stepper.cones_outside_nbhd .= true

    tau_temp = kap_temp = taukap_temp = mu_temp = 0.0
    num_pred_iters = 0
    while num_pred_iters < 100
        num_pred_iters += 1

        @. z_temp = point.z + alpha * z_dir
        @. s_temp = point.s + alpha * s_dir
        tau_temp = solver.tau + alpha * tau_dir
        kap_temp = solver.kap + alpha * kap_dir
        taukap_temp = tau_temp * kap_temp
        mu_temp = (dot(s_temp, z_temp) + taukap_temp) / (1.0 + model.nu)

        if mu_temp > 0.0
            # accept primal iterate if it is inside the cone and neighborhood
            # first check incone for whichever cones were not incone last linesearch iteration
            in_cones = true
            for k in eachindex(cones)
                if stepper.cones_outside_nbhd[k]
                    cone_k = cones[k]
                    Cones.load_point(cone_k, stepper.primal_views[k])
                    if Cones.check_in_cone(cone_k)
                        stepper.cones_outside_nbhd[k] = false
                        stepper.cones_loaded[k] = true
                    else
                        in_cones = false
                        break
                    end
                else
                    stepper.cones_loaded[k] = false
                end
            end

            if in_cones
                full_nbhd_sqr = abs2(taukap_temp - mu_temp)
                if full_nbhd_sqr < abs2(mu_temp * nbhd)
                    in_nbhds = true
                    for k in eachindex(cones)
                        cone_k = cones[k]
                        if !stepper.cones_loaded[k]
                            Cones.load_point(cone_k, stepper.primal_views[k])
                            if !Cones.check_in_cone(cone_k)
                                in_nbhds = false
                                break
                            end
                        end

                        # modifies dual_views
                        stepper.dual_views[k] .+= mu_temp .* Cones.grad(cone_k)
                        # TODO use cholesky L
                        mul!(stepper.nbhd_temp[k], Cones.inv_hess(cone_k), stepper.dual_views[k])
                        nbhd_sqr_k = dot(stepper.dual_views[k], stepper.nbhd_temp[k])

                        if nbhd_sqr_k <= -1e-5
                            println("numerical issue for cone: nbhd_sqr_k is $nbhd_sqr_k")
                            in_nbhds = false
                            break
                        elseif nbhd_sqr_k > 0.0
                            full_nbhd_sqr += nbhd_sqr_k
                            if full_nbhd_sqr > abs2(mu_temp * nbhd)
                                in_nbhds = false
                                break
                            end
                        end
                    end
                    if in_nbhds
                        break
                    end
                end
            end
        end

        if alpha < 1e-3
            # alpha is very small so just let it be zero
            return 0.0
        end

        # iterate is outside the neighborhood: decrease alpha
        alpha *= 0.8 # TODO option for parameter
    end

    return alpha
end
