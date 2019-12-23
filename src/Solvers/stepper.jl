#=
Copyright 2019, Chris Coey and contributors

interior point stepping routines for algorithms based on homogeneous self dual embedding
=#

mutable struct CombinedStepper{T <: Real} <: Stepper{T}
    in_affine_phase::Bool
    gamma::T
    rhs::Vector{T}
    x_rhs
    y_rhs
    z_rhs
    s_rhs
    s_rhs_k
    dir::Vector{T}
    x_dir
    y_dir
    z_dir
    dual_dir_k
    s_dir
    primal_dir_k
    dir_temp::Vector{T}
    res::Vector{T}
    x_res
    y_res
    z_res
    s_res
    s_res_k
    tau_row::Int
    kap_row::Int
    CombinedStepper{T}() where {T <: Real} = new{T}()
end

# create the stepper cache
function load(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    (n, p, q) = (model.n, model.p, model.q)
    cones = model.cones
    cone_idxs = model.cone_idxs

    dim = n + p + 2q + 2
    rhs = zeros(T, dim)
    dir = zeros(T, dim)
    res = zeros(T, dim)
    stepper.rhs = rhs
    stepper.dir = dir
    stepper.dir_temp = zeros(T, dim)
    stepper.res = res

    rows = 1:n
    stepper.x_rhs = view(rhs, rows)
    stepper.x_dir = view(dir, rows)
    stepper.x_res = view(res, rows)

    rows = n .+ (1:p)
    stepper.y_rhs = view(rhs, rows)
    stepper.y_dir = view(dir, rows)
    stepper.y_res = view(res, rows)

    rows = (n + p) .+ (1:q)
    stepper.z_rhs = view(rhs, rows)
    stepper.z_dir = view(dir, rows)
    stepper.z_res = view(res, rows)

    tau_row = n + p + q + 1
    stepper.tau_row = tau_row

    rows = tau_row .+ (1:q)
    stepper.s_rhs = view(rhs, rows)
    stepper.s_rhs_k = [view(rhs, tau_row .+ idxs_k) for idxs_k in cone_idxs]
    stepper.s_dir = view(dir, rows)
    stepper.s_res = view(res, rows)
    stepper.s_res_k = [view(res, tau_row .+ idxs_k) for idxs_k in cone_idxs]

    stepper.primal_dir_k = similar(stepper.s_res_k)
    stepper.dual_dir_k = similar(stepper.s_res_k)
    for (k, idxs_k) in enumerate(cone_idxs)
        s_k = view(dir, tau_row .+ idxs_k)
        z_k = view(dir, (n + p) .+ idxs_k)
        (stepper.primal_dir_k[k], stepper.dual_dir_k[k]) = (Cones.use_dual(cones[k]) ? (z_k, s_k) : (s_k, z_k))
    end

    stepper.kap_row = dim

    return stepper
end

function step(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    point = solver.point

    # @timeit solver.timer "update_fact" update_fact(solver.system_solver, solver)
    #
    # # calculate affine/prediction and combined directions
    # # TODO cleanup booleans
    # stepper.in_affine_phase = true
    # @timeit solver.timer "aff_dir" get_directions(stepper, solver)
    # z_pred = copy(stepper.z_dir)
    # s_pred = copy(stepper.s_dir)
    # tau_pred = stepper.dir[stepper.tau_row]
    # kap_pred = stepper.dir[stepper.kap_row]
    # stepper.in_affine_phase = false
    # @timeit solver.timer "comb_dir" get_directions(stepper, solver)
    # stepper.in_affine_phase = true
    #
    # # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    # solver.prev_aff_alpha = aff_alpha = find_max_alpha(stepper, solver)
    # @timeit solver.timer "aff_alpha" solver.prev_aff_alpha = aff_alpha = find_max_alpha(
    #     z_pred, s_pred, tau_pred, kap_pred, solver,
    #     nbhd = T(Inf), prev_alpha = max(solver.prev_aff_alpha, T(2e-2)), min_alpha = T(2e-2)) # TODO
    #     # nbhd = one(T), prev_alpha = max(solver.prev_aff_alpha, T(2e-2)), min_alpha = T(2e-2))
    # @assert 0 <= aff_alpha <= 1
    # # TODO allow different function (heuristic)
    # # gamma = abs2(1 - aff_alpha)
    # # gamma = (1 - aff_alpha) * min(abs2(1 - aff_alpha), T(0.25)) # from MOSEK paper
    # solver.prev_gamma = stepper.gamma = gamma
    #
    # # find distance alpha for stepping in combined direction
    # solver.prev_alpha = alpha = T(0.99) * find_max_alpha(stepper, solver) # TODO make the constant an option, depends on eps(T)?
    # if iszero(alpha)
    #     @warn("numerical failure: could not step in combined direction; terminating")
    #     solver.status = :NumericalFailure
    #     solver.keep_iterating = false
    #     return point
    # end
    #
    # # step distance alpha in combined directions
    # # TODO allow stepping different alphas in primal and dual cone directions? some solvers do
    # @. point.x += alpha * stepper.x_dir
    # @. point.y += alpha * stepper.y_dir
    # @. point.z += alpha * stepper.z_dir
    # @. point.s += alpha * stepper.s_dir
    # solver.tau += alpha * stepper.dir[stepper.tau_row]
    # solver.kap += alpha * stepper.dir[stepper.kap_row]
    # calc_mu(solver)
    #
    # if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
    #     @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
    #     solver.status = :NumericalFailure
    #     solver.keep_iterating = false
    # end



    # # calculate affine/prediction and correction directions
    # @timeit solver.timer "directions" dirs = get_directions(stepper, solver)
    # (tau_pred, tau_corr, kap_pred, kap_corr) = (stepper.tau_dirs[1], stepper.tau_dirs[2], stepper.kap_dirs[1], stepper.kap_dirs[2])

    @timeit solver.timer "update_fact" update_fact(solver.system_solver, solver)

    # calculate affine/prediction and combined directions
    # TODO cleanup booleans
    stepper.in_affine_phase = true
    @timeit solver.timer "aff_dir" get_directions(stepper, solver)
    z_pred = copy(stepper.z_dir) # TODO prealloc, cleanup
    s_pred = copy(stepper.s_dir)
    x_pred = copy(stepper.x_dir) # TODO prealloc, cleanup
    y_pred = copy(stepper.y_dir)
    tau_pred = stepper.dir[stepper.tau_row]
    kap_pred = stepper.dir[stepper.kap_row]
    stepper.in_affine_phase = false
    @timeit solver.timer "comb_dir" get_directions(stepper, solver)
    stepper.in_affine_phase = true

    # calculate correction factor gamma by finding distance aff_alpha for stepping in affine direction
    # TODO try setting nbhd to T(Inf) and avoiding the neighborhood checks - requires tuning
    @timeit solver.timer "aff_alpha" aff_alpha = find_max_alpha(
        z_pred, s_pred, tau_pred, kap_pred, solver,
        # nbhd = T(Inf), prev_alpha = max(solver.prev_aff_alpha, T(2e-2)), min_alpha = T(2e-2)) # TODO
        nbhd = one(T), prev_alpha = max(solver.prev_aff_alpha, T(2e-2)), min_alpha = T(2e-2))
    solver.prev_aff_alpha = aff_alpha
    # gamma = (1 - aff_alpha) * min(abs2(1 - aff_alpha), T(0.25))
    gamma = abs2(one(T) - aff_alpha) # TODO allow different function (heuristic)
    solver.prev_gamma = gamma

    # find distance alpha for stepping in combined direction
    z_comb = z_pred
    s_comb = s_pred
    pred_factor = one(T) - gamma
    @. z_comb = pred_factor * z_comb + gamma * stepper.z_dir
    @. s_comb = pred_factor * s_comb + gamma * stepper.s_dir
    tau_comb = pred_factor * tau_pred + gamma * stepper.dir[stepper.tau_row]
    kap_comb = pred_factor * kap_pred + gamma * stepper.dir[stepper.kap_row]
    @timeit solver.timer "comb_alpha" alpha = find_max_alpha(
        z_comb, s_comb, tau_comb, kap_comb, solver,
        nbhd = solver.max_nbhd, prev_alpha = solver.prev_alpha, min_alpha = T(1e-2))

    if iszero(alpha)
        # could not step far in combined direction, so perform a pure correction step
        solver.verbose && println("performing correction step")
        z_comb = stepper.z_dir
        s_comb = stepper.s_dir
        tau_comb = stepper.dir[stepper.tau_row]
        kap_comb = stepper.dir[stepper.kap_row]
        @timeit solver.timer "corr_alpha" alpha = find_max_alpha(
            z_comb, s_comb, tau_comb, kap_comb, solver,
            nbhd = solver.max_nbhd, prev_alpha = one(T), min_alpha = T(1e-6))

        if iszero(alpha)
            @warn("numerical failure: could not step in correction direction; terminating")
            solver.status = :NumericalFailure
            solver.keep_iterating = false
            return point
        end
        @. point.x += alpha * stepper.x_dir
        @. point.y += alpha * stepper.y_dir
    else
        @. point.x += alpha * (pred_factor * x_pred + gamma * stepper.x_dir)
        @. point.y += alpha * (pred_factor * y_pred + gamma * stepper.y_dir)
    end
    solver.prev_alpha = alpha

    # step distance alpha in combined direction
    @. point.z += alpha * z_comb
    @. point.s += alpha * s_comb
    solver.tau += alpha * tau_comb
    solver.kap += alpha * kap_comb
    calc_mu(solver)

    if solver.tau <= zero(T) || solver.kap <= zero(T) || solver.mu <= zero(T)
        @warn("numerical failure: tau is $(solver.tau), kappa is $(solver.kap), mu is $(solver.mu); terminating")
        solver.status = :NumericalFailure
        solver.keep_iterating = false
    end

    return point
end

# backtracking line search to find large distance to step in direction while remaining inside cones and inside a given neighborhood
function find_max_alpha(
    z_dir::AbstractVector{T},
    s_dir::AbstractVector{T},
    tau_dir::T,
    kap_dir::T,
    solver::Solver{T};
    nbhd::T,
    prev_alpha::T,
    min_alpha::T,
    ) where {T <: Real}
    point = solver.point
    model = solver.model
    z_temp = solver.z_temp
    s_temp = solver.s_temp

    alpha = min(prev_alpha * T(1.4), one(T)) # TODO option for parameter
    if kap_dir < zero(T)
        alpha = min(alpha, -solver.kap / kap_dir)
    end
    if tau_dir < zero(T)
        alpha = min(alpha, -solver.tau / tau_dir)
    end
    alpha *= T(0.9999)

    solver.cones_infeas .= true
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
            @timeit solver.timer "nbhd_check" in_nbhd = check_nbhd(mu_temp, taukap_temp, nbhd, solver)
            if in_nbhd
                break
            end
        end

        if alpha < min_alpha
            # alpha is very small so finish
            alpha = zero(T)
            break
        end

        # iterate is outside the neighborhood: decrease alpha
        alpha *= T(0.9) # TODO option for parameter
    end

    return alpha
end

function check_nbhd(
    mu_temp::T,
    taukap_temp::T,
    nbhd::T,
    solver::Solver{T},
    ) where {T <: Real}
    cones = solver.model.cones

    rhs_nbhd = mu_temp * abs2(nbhd)
    lhs_nbhd = abs2(taukap_temp - mu_temp) / mu_temp
    if lhs_nbhd >= rhs_nbhd
        return false
    end

    Cones.load_point.(cones, solver.primal_views)

    # accept primal iterate if it is inside the cone and neighborhood
    # first check inside cone for whichever cones were violated last line search iteration
    for (k, cone_k) in enumerate(cones)
        if solver.cones_infeas[k]
            Cones.reset_data(cone_k)
            if Cones.is_feas(cone_k)
                solver.cones_infeas[k] = false
                solver.cones_loaded[k] = true
            else
                return false
            end
        else
            solver.cones_loaded[k] = false
        end
    end

    if !isfinite(nbhd)
        return true
    end

    for (k, cone_k) in enumerate(cones)
        if !solver.cones_loaded[k]
            Cones.reset_data(cone_k)
            if !Cones.is_feas(cone_k)
                return false
            end
        end

        temp_k = solver.nbhd_temp[k]
        g_k = Cones.grad(cone_k)
        if hasfield(typeof(cone_k), :hess_fact_cache) && !Cones.update_hess_fact(cone_k)
            return false
        end
        @. temp_k = solver.dual_views[k] + g_k * mu_temp

        # TODO optionally could use multiple nbhd checks (add separate bool options and separate max_nbhd value options), eg smaller hess nbhd for each cone and larger hess nbhd for sum of cone nbhds
        if solver.use_infty_nbhd
            nbhd_k = abs2(norm(temp_k, Inf) / norm(g_k, Inf)) / mu_temp
            # nbhd_k = abs2(maximum(abs(dj) / abs(gj) for (dj, gj) in zip(duals_k, g_k))) # TODO try this neighborhood
            lhs_nbhd = max(lhs_nbhd, nbhd_k)
        else
            temp2_k = similar(temp_k) # TODO prealloc
            # TODO use dispatch
            if hasfield(typeof(cone_k), :hess_fact_cache) && cone_k.hess_fact_cache isa DenseSymCache{T}
                Cones.inv_hess_prod!(temp2_k, temp_k, cone_k)
                nbhd_k = dot(temp_k, temp2_k) / mu_temp
                if nbhd_k <= -cbrt(eps(T))
                    @warn("numerical failure: cone neighborhood is $nbhd_k")
                    return false
                end
                nbhd_k = abs(nbhd_k)
            else
                Cones.inv_hess_sqrt_prod!(temp2_k, temp_k, cone_k)
                nbhd_k = sum(abs2, temp2_k) / mu_temp
            end
            lhs_nbhd += nbhd_k
        end

        if lhs_nbhd > rhs_nbhd
            return false
        end
    end

    return true
end

# return affine or combined directions, depending on stepper.in_affine_phase
# TODO try to refactor the iterative refinement
function get_directions(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs
    dir = stepper.dir
    dir_temp = stepper.dir_temp
    res = stepper.res
    system_solver = solver.system_solver

    @timeit solver.timer "update_rhs" update_rhs(stepper, solver) # different for affine vs combined phases
    @timeit solver.timer "solve_system" solve_system(system_solver, solver, dir, rhs)

    # use iterative refinement - note apply_lhs is different for affine vs combined phases
    iter_ref_steps = (stepper.in_affine_phase ? 1 : 4) # TODO handle, maybe change dynamically, try fewer for affine phase

    copyto!(dir_temp, dir)
    res = apply_lhs(stepper, solver) # modifies res
    res .-= rhs
    norm_inf = norm(res, Inf)
    norm_2 = norm(res, 2)
    for i in 1:iter_ref_steps
        if norm_inf < 100 * eps(T) # TODO change tolerance dynamically
            break
        end
        @timeit solver.timer "solve_system" solve_system(system_solver, solver, dir, res)
        axpby!(true, dir_temp, -1, dir)
        res = apply_lhs(stepper, solver) # modifies res
        res .-= rhs

        norm_inf_new = norm(res, Inf)
        norm_2_new = norm(res, 2)
        if norm_inf_new > norm_inf || norm_2_new > norm_2
            # residual has not improved
            copyto!(dir, dir_temp)
            break
        end

        # residual has improved, so use the iterative refinement
        solver.verbose && @printf("iter ref round %d norms: inf %9.2e to %9.2e, two %9.2e to %9.2e\n", i, norm_inf, norm_inf_new, norm_2, norm_2_new)
        copyto!(dir_temp, dir)
        norm_inf = norm_inf_new
        norm_2 = norm_2_new
    end

    return dir
end

# update the RHS
function update_rhs(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    rhs = stepper.rhs

    if stepper.in_affine_phase
        # x, y, z, tau rhs
        stepper.x_rhs .= solver.x_residual
        stepper.y_rhs .= solver.y_residual
        stepper.z_rhs .= solver.z_residual
        rhs[stepper.tau_row] = solver.kap + solver.primal_obj_t - solver.dual_obj_t

        # s rhs
        for k in eachindex(solver.model.cones)
            duals_k = solver.point.dual_views[k]
            @. stepper.s_rhs_k[k] = -duals_k
        end

        # kap rhs
        rhs[end] = -solver.kap
        # TODO NT: -solver.tau
    else
        # x, y, z, tau rhs
        stepper.rhs[1:stepper.tau_row] .= 0

        # s rhs
        for (k, cone_k) in enumerate(solver.model.cones)
            grad_k = Cones.grad(cone_k)
            @. stepper.s_rhs_k[k] -= solver.mu * grad_k
            # TODO 3-order corrector?
            # if Cones.use_3order_corr(cone_k)
            #     # TODO check math here for case of cone.use_dual true - should s and z be swapped then?
            #     stepper.s_rhs_k[k] .-= Cones.correction(cone_k, stepper.primal_dir_k[k], stepper.dual_dir_k[k])
            # end
        end

        # kap rhs
        rhs[end] += solver.mu / solver.tau
        # TODO NT rhs[end] += (solver.mu - stepper.dir[stepper.tau_row] * stepper.dir[stepper.kap_row]) / solver.kap
    end

    return rhs
end

# calculate residual on 6x6 linear system
function apply_lhs(stepper::CombinedStepper{T}, solver::Solver{T}) where {T <: Real}
    model = solver.model
    tau_dir = stepper.dir[stepper.tau_row]
    kap_dir = stepper.dir[stepper.kap_row]

    # A'*y + G'*z + c*tau
    copyto!(stepper.x_res, model.c)
    mul!(stepper.x_res, model.G', stepper.z_dir, true, tau_dir)
    # -G*x + h*tau - s
    @. stepper.z_res = model.h * tau_dir - stepper.s_dir
    mul!(stepper.z_res, model.G, stepper.x_dir, -1, true)
    # -c'*x - b'*y - h'*z - kap
    stepper.res[stepper.tau_row] = -dot(model.c, stepper.x_dir) - dot(model.h, stepper.z_dir) - kap_dir
    # if p = 0, ignore A, b, y
    if !iszero(model.p)
        # A'*y + G'*z + c*tau
        mul!(stepper.x_res, model.A', stepper.y_dir, true, true)
        # -A*x + b*tau
        copyto!(stepper.y_res, model.b)
        mul!(stepper.y_res, model.A, stepper.x_dir, -1, tau_dir)
        # -c'*x - b'*y - h'*z - kap
        stepper.res[stepper.tau_row] -= dot(model.b, stepper.y_dir)
    end

    # s
    for (k, cone_k) in enumerate(model.cones)
        # (du bar) mu*H_k*z_k + s_k
        # (pr bar) z_k + mu*H_k*s_k
        s_res_k = stepper.s_res_k[k]
        Cones.hess_prod!(s_res_k, stepper.primal_dir_k[k], cone_k)
        lmul!(solver.mu, s_res_k)
        @. s_res_k += stepper.dual_dir_k[k]
    end

    # mu / (taubar^2) * tau + kap
    stepper.res[stepper.kap_row] = solver.mu / solver.tau * tau_dir / solver.tau + kap_dir
    # TODO NT: tau + taubar / kapbar * kap
    # stepper.res[stepper.kap_row] = tau_dir + solver.tau / solver.kap * kap_dir

    return stepper.res
end

# # TODO experimental for BlockMatrix LHS: if block is a Cone then define mul as hessian product, if block is solver then define mul by mu/tau/tau
# # TODO optimize... maybe need for each cone a 5-arg hess prod
# import LinearAlgebra.mul!
#
# function mul!(y::AbstractVecOrMat{T}, A::Cones.Cone{T}, x::AbstractVecOrMat{T}, alpha::Number, beta::Number) where {T <: Real}
#     # TODO in-place
#     ytemp = y * beta
#     Cones.hess_prod!(y, x, A)
#     rmul!(y, alpha)
#     y .+= ytemp
#     return y
# end
#
# function mul!(y::AbstractVecOrMat{T}, solver::Solvers.Solver{T}, x::AbstractVecOrMat{T}, alpha::Number, beta::Number) where {T <: Real}
#     rmul!(y, beta)
#     @. y += alpha * x / solver.tau * solver.mu / solver.tau
#     return y
# end
