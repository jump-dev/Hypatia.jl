
mutable struct CombinedHSDStepper <: HSDStepper
    system_solver::CombinedHSDSystemSolver
    max_nbhd::Float64
    prev_alpha::Float64
    prev_gamma::Float64

    function CombinedHSDStepper(
        model::Models.LinearModel;
        system_solver::CombinedHSDSystemSolver = (model isa Models.PreprocessedLinearModel ? QRCholCombinedHSDSystemSolver(model) : NaiveCombinedHSDSystemSolver(model)),
        max_nbhd::Float64 = 0.75,
        )
        stepper = new()
        stepper.system_solver = system_solver
        stepper.max_nbhd = max_nbhd
        stepper.prev_alpha = NaN
        stepper.prev_gamma = NaN
        return stepper
    end
end

function combined_predict_correct(solver::HSDSolver, stepper::CombinedHSDStepper)
    model = solver.model
    point = solver.point

    # calculate affine/prediction and correction directions
    (x_dirs, y_dirs, z_dirs, s_dirs, tau_dirs, kap_dirs) = get_combined_directions(solver, stepper.system_solver)

    # calculate correction factor gamma by finding distance affine_alpha for stepping in affine direction
    affine_alpha = find_max_alpha_in_nbhd(z_dirs[:, 1], s_dirs[:, 1], tau_dirs[1], kap_dirs[1], 0.999, solver)
    gamma = (1.0 - affine_alpha)^3 # TODO allow different function (heuristic)

    # find distance alpha for stepping in combined direction
    comb_scaling = [1.0 - gamma, gamma]
    z_comb = z_dirs * comb_scaling
    s_comb = s_dirs * comb_scaling
    tau_comb = dot(tau_dirs, comb_scaling)
    kap_comb = dot(kap_dirs, comb_scaling)
    alpha = find_max_alpha_in_nbhd(z_comb, s_comb, tau_comb, kap_comb, stepper.max_nbhd, solver)

    if iszero(alpha)
        # could not step far in combined direction, so perform a pure correction step
        alpha = 0.999 # TODO assumes this maintains feasibility
        comb_scaling = [0.0, 1.0]
        z_comb = z_dirs * comb_scaling
        s_comb = s_dirs * comb_scaling
        tau_comb = dot(tau_dirs, comb_scaling)
        kap_comb = dot(kap_dirs, comb_scaling)
    end

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

    stepper.prev_gamma = gamma
    stepper.prev_alpha = alpha

    return point
end

function print_iter_header(solver::HSDSolver, stepper::CombinedHSDStepper)
    @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s %9s %9s %9s\n",
        "iter", "p_obj", "d_obj", "abs_gap", "rel_gap",
        "x_feas", "y_feas", "z_feas", "tau", "kap", "mu",
        "gamma", "alpha",
        )
    flush(stdout)
end

function print_iter_summary(solver::HSDSolver, stepper::CombinedHSDStepper)
    @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
        solver.num_iters, solver.primal_obj, solver.dual_obj, solver.gap, solver.rel_gap,
        solver.x_feas, solver.y_feas, solver.z_feas, solver.tau, solver.kap, solver.mu,
        stepper.prev_gamma, stepper.prev_alpha,
        )
    flush(stdout)
end
