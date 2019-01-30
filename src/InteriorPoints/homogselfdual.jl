#=
Copyright 2018, Chris Coey and contributors

interior point type and functions for algorithms based on homogeneous self dual embedding

TODO make internal statuses types
=#

# TODO maybe have point defined in model, and keep tau and kappa outside the point in hsde alg

mutable struct HSDEPoint <: InteriorPoint
    tx::Vector{Float64}
    ty::Vector{Float64}
    tz::Vector{Float64}
    ts::Vector{Float64}
    tau::Float64
    kap::Float64
    # mu::Float64
    tz_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}
    ts_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}
    dual_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}
    primal_views::Vector{SubArray{Float64,1,Vector{Float64},Tuple{UnitRange{Int}},true}}

    function HSDEPoint(model::Models.LinearObjConic)
        point = new()
        point.tx = zeros(length(model.c))
        point.ty = zeros(length(model.b))
        point.tz = zeros(length(model.h))
        point.ts = zeros(length(model.h))
        point.tau = 1.0
        point.kap = 1.0
        # point.mu = 1.0
        point.tz_views = [view(point.tz, idxs) for idxs in model.cone_idxs]
        point.ts_views = [view(point.ts, idxs) for idxs in model.cone_idxs]
        point.dual_views = [Cones.use_dual(model.cones[k]) ? point.ts_views[k] : point.tz_views[k] for k in eachindex(model.cones)]
        point.primal_views = [Cones.use_dual(model.cones[k]) ? point.tz_views[k] : point.ts_views[k] for k in eachindex(model.cones)]
        return point
    end
end

function unscale_point(point::HSDEPoint)
    point.tx ./= point.tau
    point.ty ./= point.tau
    point.tz ./= point.tau
    point.ts ./= point.tau
    return point
end

mutable struct HSDESolver <: IPMSolver
    model::Models.LinearObjConic # TODO the cone LP type model data
    # linear_solver::LinearSystems.LinearSystemSolver

    # options
    verbose::Bool
    tol_rel_opt::Float64
    tol_abs_opt::Float64
    tol_feas::Float64
    max_iters::Int
    time_limit::Float64
    combined_nbhd::Float64

    # point
    point::HSDEPoint
    # aux_point::HSDEPoint

    converge_tol_x::Float64
    converge_tol_y::Float64
    converge_tol_z::Float64

    residual_x::Vector{Float64}
    residual_y::Vector{Float64}
    residual_z::Vector{Float64}

    # solve info
    status::Symbol
    num_iters
    solve_time
    primal_obj
    dual_obj

    function HSDESolver(
        model::Models.LinearObjConic,
        # linear_solver::LinearSystems.LinearSystemSolver,
        ;
        verbose::Bool = true,
        tol_rel_opt = 1e-6,
        tol_abs_opt = 1e-7,
        tol_feas = 1e-7,
        max_iters::Int = 100,
        time_limit::Float64 = 3e2,
        combined_nbhd::Float64 = 0.1,
        )
        solver = new()
        solver.model = model
        # solver.linear_solver = linear_solver

        solver.verbose = verbose
        solver.tol_rel_opt = tol_rel_opt
        solver.tol_abs_opt = tol_abs_opt
        solver.tol_feas = tol_feas
        solver.max_iters = max_iters
        solver.time_limit = time_limit
        solver.combined_nbhd = combined_nbhd

        solver.point = get_initial_point(solver.model)

        # solver.prediction_dir = HSDEPoint(model)
        # solver.correction_dir = HSDEPoint(model)

        solver.converge_tol_x = _get_tol(model.c)
        solver.converge_tol_y = _get_tol(model.b)
        solver.converge_tol_z = _get_tol(model.h)

        solver.residual_x = similar(model.c)
        solver.residual_y = similar(model.b)
        solver.residual_z = similar(model.h)

        solver.status = :SolveNotCalled
        return solver
    end
end


get_tau(solver::HSDESolver) = solver.point.tau
get_kappa(solver::HSDESolver) = solver.point.kappa
get_mu(solver::HSDESolver) = solver.point.mu


function solve(solver::HSDESolver)
    solver.status = :SolveCalled
    start_time = time()

    solver.verbose && @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s\n",
            "iter", "p_obj", "d_obj", "abs_gap", "rel_gap", "p_inf", "d_inf", "tau", "kap", "mu")

    solver.num_iters = 0
    while check_convergence(solver)
        if solver.num_iters == solver.max_iters
            solver.verbose && println("iteration limit reached; terminating")
            solver.status = :IterationLimit
            break
        end

        if time() - start_time >= solver.time_limit
            solver.verbose && println("time limit reached; terminating")
            solver.status = :TimeLimit
            break
        end

        # TODO may use different function, or function could change during some iteration eg if numerical difficulties
        (point, mu) = combined_predict_correct(point, residual, mu, solver)

        solver.num_iters += 1
    end

    # calculate result and iteration statistics and finish
    unscale_point(solver.point)
    solver.solve_time = time() - start_time

    if solver.verbose
        println("\nstatus is $(solver.status) after $(solver.num_iters) iterations and $(trunc(solver.solve_time, digits=3)) seconds\n")
    end

    return
end

function check_convergence(solver::HSDESolver)
    # TODO maybe instead store all the convergence details of the point inside the point
    (norm_res_x, norm_res_tx) = _get_residual_x(residual_x, point, model)
    (norm_res_y, norm_res_ty) = _get_residual_y(residual_y, point, model)
    (norm_res_z, norm_res_tz) = _get_residual_z(residual_z, point, model)
    norm_res_primal = max(norm_res_ty * converge_tol_y, norm_res_tz * converge_tol_z)
    norm_res_dual = norm_res_tx * converge_tol_x

    obj_t_primal = dot(model.c, point.tx)
    obj_t_dual = -dot(model.b, point.ty) - dot(model.h, point.tz)
    obj_primal = obj_t_primal / point.tau
    obj_dual = obj_t_dual / point.tau
    gap = dot(point.tz, point.ts) # TODO maybe should adapt original Alfonso condition instead of using this CVXOPT condition
    rel_gap = _get_rel_gap(obj_primal, obj_dual, gap)

    # print iteration statistics
    if solver.verbose
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            solver.num_iters, obj_primal, obj_dual, gap, rel_gap,
            norm_res_primal, norm_res_dual, point.tau, point.kap, mu
            )
        flush(stdout)
    end

    # check convergence criteria
    # TODO nearly primal or dual infeasible or nearly optimal cases?
    if norm_res_primal <= solver.tol_feas && norm_res_dual <= solver.tol_feas &&
        (gap <= solver.tol_abs_opt || (!isnan(rel_gap) && rel_gap <= solver.tol_rel_opt))
        solver.verbose && println("optimal solution found; terminating")
        solver.status = :Optimal
        return false
    end
    if obj_t_dual > 0.0
        infres_pr = norm_res_x * converge_tol_x / obj_t_dual
        if infres_pr <= solver.tol_feas
            solver.verbose && println("primal infeasibility detected; terminating")
            solver.status = :PrimalInfeasible
            return false
        end
    end
    if obj_t_primal < 0.0
        infres_du = -max(norm_res_y * converge_tol_y, norm_res_z * converge_tol_z) / obj_t_primal
        if infres_du <= solver.tol_feas
            solver.verbose && println("dual infeasibility detected; terminating")
            solver.status = :DualInfeasible
            return false
        end
    end
    if mu <= solver.tol_feas * 1e-2 && point.tau <= solver.tol_feas * 1e-2 * min(1.0, point.kap)
        solver.verbose && println("ill-posedness detected; terminating")
        solver.status = :IllPosed
        return false
    end

    return true
end


function combined_predict_correct(point::HSDEPoint, residual::HSDEPoint, mu::Float64, solver::HSDESolver)
    cones = solver.model.cones
    (n, p, q) = (model.n, model.p, model.q)

    # calculate prediction and correction directions

    LHS[n+p+q+1, end] = mu / point.tau / point.tau
    for k in eachindex(cones)
        cone_k = cones[k]
        # TODO stepped to this point so should already have called check_in_cone for the point
        Cones.load_point(cone_k, point.primal_views[k])
        @assert Cones.check_in_cone(cone_k)
        rows = (n + p) .+ model.cone_idxs[k]
        cols = Cones.use_dual(cone_k) ? rows : (q + 1) .+ rows
        LHS[rows, cols] = mu * Cones.hess(cone_k)
    end

    rhs = [
        residual_x  zeros(n);
        residual_y  zeros(p);
        zeros(q)    zeros(q);
        -point.kap  -point.kap + mu / point.tau;
        residual_z  zeros(q);
        point.kap + obj_t_primal - obj_t_dual  0.0;
        ]
    for k in eachindex(cones)
        rows = (n + p) .+ model.cone_idxs[k]
        rhs[rows, 1] = -point.dual_views[k]
        rhs[rows, 2] = -point.dual_views[k] - mu * Cones.grad(cones[k])
    end

    F = lu(LHS)
    ldiv!(F, rhs)

    # affine phase
    # affine_direction = construct_affine_direction(direction_solution, mu, solver)
    @. @views begin
        predict.tx = rhs[1:n, 1]
        predict.ty = rhs[(n + 1):(n + p), 1]
        predict.tz = rhs[(n + p + 1):(n + p + q), 1]
        predict.ts = rhs[(n + p + q + 2):(n + p + 2q + 1), 1]
    end
    predict.kap = rhs[n + p + q + 1, 1]
    predict.tau = rhs[n + p + 2q + 2, 1]

    # affine_alpha = get_max_alpha(point, predict, solver)
    affine_alpha = get_max_alpha_in_nbhd(point, predict, mu, 0.99, solver)


    # # NOTE step in corrector direction here: not in description of algorithms?
    # @. @views begin
    #     correct.tx = rhs[1:n, 2]
    #     correct.ty = rhs[(n + 1):(n + p), 2]
    #     correct.tz = rhs[(n + p + 1):(n + p + q), 2]
    #     correct.ts = rhs[(n + p + q + 2):(n + p + 2q + 1), 2]
    # end
    # correct.kap = rhs[n + p + q + 1, 2]
    # correct.tau = rhs[n + p + 2q + 2, 2]
    #
    # point = step_in_direction(point, correct, 1.0)
    # mu = get_mu(point, model)


    # combined phase
    gamma = (1.0 - affine_alpha)^3 # TODO allow different function (heuristic)
    # @show gamma

    # direction = construct_combined_direction(direction_solution, mu, gamma, solver)
    combined_rhs = rhs * vcat(1.0 - gamma, gamma)
    combined = predict
    @. @views begin
        combined.tx = combined_rhs[1:n]
        combined.ty = combined_rhs[(n + 1):(n + p)]
        combined.tz = combined_rhs[(n + p + 1):(n + p + q)]
        combined.ts = combined_rhs[(n + p + q + 2):(n + p + 2q + 1)]
    end
    combined.kap = combined_rhs[n + p + q + 1]
    combined.tau = combined_rhs[n + p + 2q + 2]

    alpha = get_max_alpha_in_nbhd(point, combined, mu, solver.combined_nbhd, solver)

    point = step_in_direction(point, combined, alpha)
    mu = get_mu(point, model)

    return point
end

function solve_linear_system(mu, solver)

    # 3 columns:
    # 1) fixed c b h
    # 2) predictor rhs (residual)
    # 3) corrector rhs (zero)
    # TODO prealloc, also note first col and some of 3rd col don't change
    # rhs_x = [-model.c residual_x zeros(model.n)]
    # rhs_y = [-model.b residual_y zeros(model.p)]
    # rhs_z = [-model.h residual_z zeros(model.q)]

    # TODO cache
    @. yi[:, 1] = L.b
    @. yi[:, 2] = -rhs_ty
    @. xi[:, 1] = -L.c
    @. xi[:, 2] = rhs_tx
    z1 = view(zi, :, 1)
    z2 = view(zi, :, 2)




    # eliminate

    # calculate z2
    @. z2 = -rhs_tz
    for k in eachindex(L.cone.cones)
        a1k = view(z1, L.cone.idxs[k])
        a2k = view(z2, L.cone.idxs[k])
        a3k = view(rhs_ts, L.cone.idxs[k])
        if L.cone.cones[k].use_dual
            @. a1k = a2k - a3k
            Cones.calcHiarr!(a2k, a1k, L.cone.cones[k])
            a2k ./= mu
        elseif !iszero(a3k) # TODO rhs_ts = 0 for correction steps, so can just check if doing correction
            Cones.calcHarr!(a1k, a3k, L.cone.cones[k])
            @. a2k -= mu * a1k
        end
    end

    # calculate z1
    if iszero(L.h) # TODO can check once when creating cache
        z1 .= 0.0
    else
        for k in eachindex(L.cone.cones)
            a1k = view(L.h, L.cone.idxs[k])
            a2k = view(z1, L.cone.idxs[k])
            if L.cone.cones[k].use_dual
                Cones.calcHiarr!(a2k, a1k, L.cone.cones[k])
                a2k ./= mu
            else
                Cones.calcHarr!(a2k, a1k, L.cone.cones[k])
                a2k .*= mu
            end
        end
    end


    # call 3x3 solve routine
    (x_sol, y_sol, z_sol) = LinearSystems.solve(solver.linear_solver, x_rhs, y_rhs, z_rhs)




    # reconstruct using matrix operations
    dir_tau = (rhs_tau + rhs_kap + dot(L.c, xi[:, 2]) + dot(L.b, yi[:, 2]) + dot(L.h, z2)) /
        (mu / tau / tau - dot(L.c, xi[:, 1]) - dot(L.b, yi[:, 1]) - dot(L.h, z1))

    rhs_tx = xi[:, 2] + dir_tau * xi[:, 1]
    rhs_ty = yi[:, 2] + dir_tau * yi[:, 1]
    rhs_tz = z2 + dir_tau * z1

    mul!(z1, L.G, rhs_tx)
    @. rhs_ts = -z1 + L.h * dir_tau - rhs_ts
    dir_kap = -dot(L.c, rhs_tx) - dot(L.b, rhs_ty) - dot(L.h, rhs_tz) - rhs_tau






    # combine for full prediction and correction directions
    # TODO maybe prealloc the views
    x_fix = view(x_sol, :, 1)
    y_fix = view(y_sol, :, 1)
    z_fix = view(z_sol, :, 1)

    x_pred = view(x_sol, :, 2)
    y_pred = view(y_sol, :, 2)
    z_pred = view(z_sol, :, 2)
    prediction_direction = construct_direction(prediction_direction, x_fix, y_fix, z_fix, x_pred, y_pred, z_pred, solver)

    x_corr = view(x_sol, :, 3)
    y_corr = view(y_sol, :, 3)
    z_corr = view(z_sol, :, 3)
    correction_direction = construct_direction(correction_direction, x_fix, y_fix, z_fix, x_corr, y_corr, z_corr, solver)





    # combine for correction direction



    return (prediction_direction, correction_direction)
end

function construct_direction(direction, x_fix, y_fix, z_fix, x_var, y_var, z_var, solver)
    model = solver.model

    direction.tau = (rhs_tau + rhs_kap + dot(L.c, xi[:, 2]) + dot(L.b, yi[:, 2]) + dot(L.h, z2)) /
        (mu / tau / tau - dot(L.c, xi[:, 1]) - dot(L.b, yi[:, 1]) - dot(L.h, z1))
    @. @views rhs_tx = xi[:, 2] + dir_tau * xi[:, 1]
    @. @views rhs_ty = yi[:, 2] + dir_tau * yi[:, 1]
    @. rhs_tz = z2 + dir_tau * z1

    mul!(z1, L.G, rhs_tx)
    @. rhs_ts = -z1 + L.h * dir_tau - rhs_ts
    dir_kap = -dot(L.c, rhs_tx) - dot(L.b, rhs_ty) - dot(L.h, rhs_tz) - rhs_tau

    return direction
end



_get_tol(v::Vector{Float64}) = inv(max(1.0, norm(v)))

function _get_rel_gap(obj_primal::Float64, obj_dual::Float64, gap::Float64)
    if obj_primal < 0.0
        return gap / -obj_primal
    elseif obj_dual > 0.0
        return gap / obj_dual
    else
        return NaN
    end
end

function _get_residual_x(residual_x::Vector{Float64}, point::HSDEPoint, model::Models.LinearObjConic)
    # residual_x = -A'*ty - G'*tz - c*tau
    residual_x .= -model.A' * point.ty - model.G' * point.tz # TODO remove allocs
    norm_res_x = norm(residual_x)
    @. residual_x -= model.c * point.tau
    norm_res_tx = norm(residual_x) / point.tau
    return (norm_res_x, norm_res_tx)
end

function _get_residual_y(residual_y::Vector{Float64}, point::HSDEPoint, model::Models.LinearObjConic)
    # residual_y = A*tx - b*tau
    mul!(residual_y, model.A, point.tx)
    norm_res_y = norm(residual_y)
    @. residual_y -= model.b * point.tau
    norm_res_ty = norm(residual_y) / point.tau
    return (norm_res_y, norm_res_ty)
end

function _get_residual_z(residual_z::Vector{Float64}, point::HSDEPoint, model::Models.LinearObjConic)
    # residual_z = ts + G*tx - h*tau
    mul!(residual_z, model.G, point.tx)
    @. residual_z += point.ts
    norm_res_z = norm(residual_z)
    @. residual_z -= model.h * point.tau
    norm_res_tz = norm(residual_z) / point.tau
    return (norm_res_z, norm_res_tz)
end

get_mu(point::HSDEPoint, model::Models.LinearObjConic) =
    (dot(point.tz, point.ts) + point.tau * point.kap) / (1.0 + model.nu)

function get_initial_point(model::Models.LinearObjConic)
    point = HSDEPoint(model)

    cones = model.cones
    for k in eachindex(cones)
        cone_k = cones[k]
        primal_k = point.primal_views[k]
        Cones.set_initial_point(primal_k, cone_k)
        Cones.load_point(cone_k, primal_k)
        @assert Cones.check_in_cone(cone_k)
        point.dual_views[k] .= -Cones.grad(cone_k)
    end

    # solve for ty
    # A'y = -c - G'z
    # solve for tx
    # Ax = b
    # Gx = h - ts
    temp_n = -model.c - model.G' * point.tz # TODO remove allocs
    temp_p_q = vcat(model.b, model.h - point.ts) # TODO remove allocs
    # if has_QR(model) # TODO reuse QR from preprocessing
    #     ldiv!(point.ty, model.At_qr, temp_n)
    #     ldiv!(point.tx, model.AG_qr, temp_p_q)
    # else
        if !isempty(point.ty)
            At_fact = factorize(issparse(model.A) ? sparse(model.A') : Matrix(model.A'))
            # ldiv!(point.ty, At_fact, temp_n)
            point.ty .= At_fact \ temp_n
        end
        AG_fact = factorize(vcat(model.A, model.G))
        # ldiv!(point.tx, AG_fact, temp_p_q)
        point.tx .= AG_fact \ temp_p_q
    # end

    solver.mu = get_mu(solver.point, solver.model)
    if isnan(solver.) || abs(1.0 - solver.mu) > 1e-6
        error("initial mu is $(solver.mu) (should be 1.0)")
    end

    return point
end

function step_in_direction(point::HSDEPoint, direction::HSDEPoint, alpha::Float64)
    @. point.tx += alpha * direction.tx
    @. point.ty += alpha * direction.ty
    @. point.tz += alpha * direction.tz
    @. point.ts += alpha * direction.ts
    point.tau += alpha * direction.tau
    point.kap += alpha * direction.kap
    return point
end

function get_max_alpha_in_nbhd(point::HSDEPoint, direction::HSDEPoint, mu::Float64, nbhd::Float64, solver::HSDESolver)
    model = solver.model
    cones = model.cones

    alpha = 1.0 # TODO maybe start at previous alpha but increased slightly, or use affine_alpha
    if direction.kap < 0.0
        alpha = min(alpha, -point.kap / direction.kap)
    end
    if direction.tau < 0.0
        alpha = min(alpha, -point.tau / direction.tau)
    end
    # TODO what about mu? quadratic equation. need dot(ls_ts, ls_tz) + ls_tau * ls_kap > 0
    alpha *= 0.99

    # ONLY BACKTRACK, NO FORWARDTRACK

    ls_tz = similar(point.tz) # TODO prealloc
    ls_ts = similar(point.ts)
    primal_views = [view(Cones.use_dual(cones[k]) ? ls_tz : ls_ts, model.cone_idxs[k]) for k in eachindex(cones)]
    dual_views = [view(Cones.use_dual(cones[k]) ? ls_ts : ls_tz, model.cone_idxs[k]) for k in eachindex(cones)]

    # cones_outside_nbhd = trues(length(cones))
    # TODO sort cones so that check the ones that failed in-cone check last iteration first

    ls_tau = ls_kap = ls_tk = ls_mu = 0.0
    num_pred_iters = 0
    while num_pred_iters < 100
        num_pred_iters += 1

        @. ls_tz = point.tz + alpha * direction.tz
        @. ls_ts = point.ts + alpha * direction.ts
        ls_tau = point.tau + alpha * direction.tau
        ls_kap = point.kap + alpha * direction.kap
        ls_tk = ls_tau * ls_kap
        ls_mu = (dot(ls_ts, ls_tz) + ls_tk) / (1.0 + model.nu)

        # accept primal iterate if
        # - decreased alpha and it is the first inside the cone and beta-neighborhood or
        # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
        # if ls_mu > 0.0 && abs(ls_tk - ls_mu) / ls_mu < nbhd # condition for 1-dim nonneg cone for tau and kap
        #     in_nbhds = true
        #     for k in eachindex(cones)
        #         cone_k = cones[k]
        #         Cones.load_point(cone_k, primal_views[k])
        #         if !Cones.check_in_cone(cone_k) || get_nbhd(cone_k, dual_views[k], ls_mu) > nbhd
        #             in_nbhds = false
        #             break
        #         end
        #     end
        #     if in_nbhds
        #         break
        #     end
        # end
        if ls_mu > 0.0
            full_nbhd_sqr = abs2(ls_tk - ls_mu)
            in_nbhds = true
            for k in eachindex(cones)
                cone_k = cones[k]
                Cones.load_point(cone_k, primal_views[k])
                if !Cones.check_in_cone(cone_k)
                    in_nbhds = false
                    break
                end
                full_nbhd_sqr += get_nbhd(cone_k, dual_views[k], ls_mu)
                if full_nbhd_sqr > abs2(ls_mu * nbhd)
                    in_nbhds = false
                    break
                end
            end
            if in_nbhds
                break
            end
        end


        # iterate is outside the neighborhood: decrease alpha
        alpha *= 0.8 # TODO option for parameter
    end

    if alpha < 1e-7 # TODO return slow progress status or find a workaround
        error("alpha is $alpha")
    end

    return alpha
end

function get_nbhd(cone::Cones.Cone, duals::AbstractVector{Float64}, mu::Float64)
    # TODO no allocs
    temp = duals + mu * Cones.grad(cone)
    # TODO use cholesky L
    # nbhd = sqrt(temp' * Cones.inv_hess(cone) * temp) / mu
    nbhd = temp' * Cones.inv_hess(cone) * temp
    # @show nbhd
    return nbhd
end


# function get_prediction_direction(point::HSDEPoint, residual::HSDEPoint, solver::HSDESolver)
#     for k in eachindex(solver.model.cones)
#         @. residual.tz_views[k] = -point.primal_views[k]
#     end
#     residual.tau = kap + cx + by + hz
#     residual.kap = -kap
#     return LinearSystems.solvelinsys6!(point, residual, mu, solver)
# end

# function get_correction_direction(point::HSDEPoint, solver::HSDESolver)
#     corr_rhs = solver.aux_point
#     @. corr_rhs.tx = 0.0
#     @. corr_rhs.ty = 0.0
#     cones = solver.model.cones
#     for k in eachindex(cones)
#         @. corr_rhs.tz_views[k] = -point.primal_views[k] - mu * Cones.grad(cones[k])
#     end
#     @. corr_rhs.ts = 0.0
#     corr_rhs.tau = 0.0
#     corr_rhs.kap = -point.kap + mu / point.tau
#     return LinearSystems.solvelinsys6!(point, corr_rhs, mu, solver)
# end

# function predict_then_correct(point::HSDEPoint, residual::HSDEPoint, mu::Float64, solver::HSDESolver)
#     cones = solver.model.cones
#
#     # prediction phase
#     direction = get_prediction_direction(point, residual, solver)
#     alpha = 0.9999 * get_max_alpha(point, direction, solver)
#     point = step_in_direction(point, direction, alpha)
#
#     # correction phase
#     num_corr_steps = 0
#     while nbhd > eta && num_corr_steps < solver.max_corr_steps
#         direction = get_correction_direction(point, solver)
#         point = step_in_direction(point, direction, 0.9999)
#         for k in eachindex(cones)
#             @assert Cones.check_in_cone(cones[k], point.primal_views[k])
#         end
#         num_corr_steps += 1
#     end
#     if num_corr_steps == solver.max_corr_steps
#         solver.verbose && println("corrector phase finished outside the eta-neighborhood; terminating")
#         solver.status = :CorrectorFail
#         return (false, point)
#     end
#
#     return (true, point)
# end





#
# # get neighborhood parameters depending on magnitude of barrier parameter and maximum number of correction steps
# # TODO calculate values from the formulae given in Papp & Yildiz "On A Homogeneous Interior-Point Algorithm for Non-Symmetric Convex Conic Optimization"
# function getbetaeta(maxcorrsteps::Int, bnu::Float64)
#     if maxcorrsteps <= 2
#         if bnu < 10.0
#             return (0.1810, 0.0733, 0.0225)
#         elseif bnu < 100.0
#             return (0.2054, 0.0806, 0.0263)
#         else
#             return (0.2190, 0.0836, 0.0288)
#         end
#     elseif maxcorrsteps <= 4
#         if bnu < 10.0
#             return (0.2084, 0.0502, 0.0328)
#         elseif bnu < 100.0
#             return (0.2356, 0.0544, 0.0380)
#         else
#             return (0.2506, 0.0558, 0.0411)
#         end
#     else
#         if bnu < 10.0
#             return (0.2387, 0.0305, 0.0429)
#         elseif bnu < 100.0
#             return (0.2683, 0.0327, 0.0489)
#         else
#             return (0.2844, 0.0332, 0.0525)
#         end
#     end
# end
