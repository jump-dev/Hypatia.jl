#=
Copyright 2018, Chris Coey and contributors

interior point type and functions for algorithms based on homogeneous self dual embedding
=#

mutable struct HSDESolver <: IPMSolver
    model::Models.LinearObjConic # TODO the cone LP type model data
    linsolver::LinearSystems.LinearSystemSolver

    status::Symbol

    # options


    # points
    point
    point2

    # result
    x
    y
    z
    s
    tau
    kappa
    mu

    # solve info
    iterations
    solve_time
    primal_obj
    dual_obj


end


mutable struct HSDEPoint <: InteriorPoint
    tx::Vector{Float64}
    ty::Vector{Float64}
    tz::Vector{Float64}
    ts::Vector{Float64}
    tau::Float64
    kap::Float64
    # TODO possibly keep mu here too, if it would save recalculating mu a few times
end
HSDEPoint(n::Int, p::Int, q::Int) = HSDEPoint(zeros(n), zeros(p), zeros(q), zeros(q), 1.0, 1.0)

function solve(solver::HSDESolver)
    solver.status = :SolveCalled
    time0 = time()

    point = get_initial_point(solver)

    iter = 0
    check_converged(point, solver, iter)

    # iterate while solve time remains
    while time() - time0 < solver.timelimit
        iter += 1

        point = predict_then_correct(point, solver) # TODO may use different function, or function could change during some iteration eg if numerical difficulties

        check_converged(point, solver, iter) && break

        if iter == solver.maxiter
            solver.verbose && println("iteration limit reached; terminating")
            solver.status = :IterationLimit
        end
    end

    # calculate result and iteration statistics
    @. solver.x = tx / tau
    @. solver.y = ty / tau
    @. solver.z = tz / tau
    @. solver.s = ts / tau
    solver.tau = tau
    solver.kap = kap
    solver.mu = mu
    solver.niters = iter
    solver.solvetime = time() - time0

    solver.verbose && println("\nstatus is $(solver.status) after $iter iterations and $(trunc(solver.solvetime, digits=3)) seconds\n")

    return
end

function check_converged(point::HSDEPoint, solver::HSDESolver, num_iters::Int)
    # TODO delete these if not so useful
    model = solver.model
    (c, A, b, G, h, cone) = (model.c, model.A, model.b, model.G, model.h, model.cone)
    (tx, ty, tz, ts) = (point.tx, point.ty, point.tz, point.ts)

    # TODO rename tol variables to reflect which constraints they refer to
    # TODO consider 3 tolerances, one for primal equalities, one for primal cone constraint, and one for dual equalities


    # TODO calculate only once. put in solver
    # calculate tolerances for convergence
    tol_res_tx = inv(max(1.0, norm(c)))
    tol_res_ty = inv(max(1.0, norm(b)))
    tol_res_tz = inv(max(1.0, norm(h)))


    # TODO preallocate. can use the unused point. so then the residual is essentially a direction
    # helper arrays for residuals, right-hand-sides, and search directions
    tmp_tx = similar(tx)
    tmp_tx2 = similar(tx) # TODO can delete once have mul-add in-place
    tmp_ty = similar(ty)
    tmp_tz = similar(tz)
    tmp_ts = similar(ts)


    # calculate residuals and convergence parameters
    # tmp_tx = -A'*ty - G'*tz - c*tau
    mul!(tmp_tx2, A', ty)
    mul!(tmp_tx, G', tz)
    @. tmp_tx = -tmp_tx2 - tmp_tx
    nres_x = norm(tmp_tx)
    @. tmp_tx -= c * tau
    nres_tx = norm(tmp_tx) / tau

    # tmp_ty = A*tx - b*tau
    mul!(tmp_ty, A, tx)
    nres_y = norm(tmp_ty)
    @. tmp_ty -= b * tau
    nres_ty = norm(tmp_ty) / tau

    # tmp_tz = ts + G*tx - h*tau
    mul!(tmp_tz, G, tx)
    @. tmp_tz += ts
    nres_z = norm(tmp_tz)
    @. tmp_tz -= h * tau
    nres_tz = norm(tmp_tz) / tau

    (cx, by, hz) = (dot(c, tx), dot(b, ty), dot(h, tz))
    obj_pr = cx / tau
    obj_du = -(by + hz) / tau
    gap = dot(tz, ts) # TODO maybe should adapt original Alfonso condition instead of using this CVXOPT condition

    # TODO maybe add small epsilon to denominators that are zero to avoid NaNs, and get rid of isnans further down
    if obj_pr < 0.0
        relgap = gap / -obj_pr
    elseif obj_du > 0.0
        relgap = gap / obj_du
    else
        relgap = NaN
    end

    nres_pr = max(nres_ty * tol_res_ty, nres_tz * tol_res_tz)
    nres_du = nres_tx * tol_res_tx

    # print iteration statistics
    if solver.verbose
        if iszero(num_iters)
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s\n",
                "iter", "p_obj", "d_obj", "abs_gap", "rel_gap", "p_inf", "d_inf", "tau", "kap", "mu")
        end
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            num_iters, obj_pr, obj_du, gap, relgap, nres_pr, nres_du, tau, kap, mu)
        flush(stdout)
    end

    # check convergence criteria
    # TODO nearly primal or dual infeasible or nearly optimal cases?
    if nres_pr <= solver.tolfeas && nres_du <= solver.tolfeas &&
        (gap <= solver.tolabsopt || (!isnan(relgap) && relgap <= solver.tolrelopt))
        solver.verbose && println("optimal solution found; terminating")
        solver.status = :Optimal
        return true
    end
    if hz + by < 0.0
        infres_pr = nres_x * tol_res_tx / (-hz - by)
        if infres_pr <= solver.tolfeas
            solver.verbose && println("primal infeasibility detected; terminating")
            solver.status = :PrimalInfeasible
            return true
        end
    end
    if cx < 0.0
        infres_du = -max(nres_y * tol_res_ty, nres_z * tol_res_tz) / cx
        if infres_du <= solver.tolfeas
            solver.verbose && println("dual infeasibility detected; terminating")
            solver.status = :DualInfeasible
            return true
        end
    end
    if mu <= solver.tolfeas * 1e-2 && tau <= solver.tolfeas * 1e-2 * min(1.0, kap)
        solver.verbose && println("ill-posedness detected; terminating")
        solver.status = :IllPosed
        return true
    end

    return false
end

get_mu(point::HSDEPoint, cone::Cones.Cone) = (dot(point.tz, point.ts) + point.tau * point.kap) / (1.0 + cone.barrier_param)

# calculate neighborhood distance to central path
function get_neighborhood(point::HSDEPoint, solver::HSDESolver)

    (g, ts, tz, mu, cone) =

    for k in eachindex(cone.prmtvs)
        calcg_prmtv!(view(g, cone.idxs[k]), cone.prmtvs[k])
        (v1, v2) = (cone.prmtvs[k].usedual ? (ts, tz) : (tz, ts))
        @. @views v1[cone.idxs[k]] += mu*g[cone.idxs[k]]
        # @. @views v1[cone.idxs[k]] += g[cone.idxs[k]]
        Cones.calcHiarr_prmtv!(view(v2, cone.idxs[k]), view(v1, cone.idxs[k]), cone.prmtvs[k])
    end
    return dot(ts, tz)
end

function get_initial_point(solver::HSDESolver)
    # TODO delete these if not so useful
    model = solver.model
    (c, A, b, G, h, cone) = (model.c, model.A, model.b, model.G, model.h, model.cone)
    (n, p, q) = (length(c), length(b), length(h))

    point = HSDEPoint(n, p, q)
    (tx, ty, tz, ts) = (point.tx, point.ty, point.tz, point.ts)

    Cones.set_central_point!(ts, tz, cone) # TODO scale like in alfonso?
    mu = get_mu(point, cone)
    if isnan(mu) || abs(1.0 - mu) > 1e-6
        error("initial mu is $mu")
    end

    # solve for ty
    # A'y = -c - G'z
    # TODO re-use factorization of A' from preprocessing
    # TODO remove allocs
    if issparse(A)
        if !isempty(ty)
            ty .= sparse(A') \ (-c - G' * tz)
        end
    else
        ty .= A' \ (-c - G' * tz)
    end

    # solve for tx
    # Ax = b
    # Gx = h - ts
    # TODO re-use factorization of [A; G] from preprocessing
    # TODO remove allocs
    tx .= [A; G] \ [b; h - ts]

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

function get_prediction_direction(point::HSDEPoint, direction::HSDEPoint, solver::HSDESolver)

    # TODO direction is another helper point (with preallocated space)

    # calculate prediction direction
    @. ls_tz = tz
    @. ls_ts = ts
    @. tmp_ts = tmp_tz
    for k in eachindex(cone.prmtvs)
        v1 = (cone.prmtvs[k].usedual ? ts : tz)
        @. @views tmp_tz[cone.idxs[k]] = -v1[cone.idxs[k]]
    end

    (tmp_kap, tmp_tau) = LinearSystems.solvelinsys6!(tmp_tx, tmp_ty, tmp_tz, -kap, tmp_ts, kap + cx + by + hz, mu, tau, L)


    return direction
end

function perform_prediction(point::HSDEPoint, point2::HSDEPoint, direction::HSDEPoint, solver::HSDESolver)

    # determine step length alpha by line search
    alpha = alphapred
    if tmp_kap < 0.0
        alpha = min(alpha, -kap / tmp_kap * 0.99999)
    end
    if tmp_tau < 0.0
        alpha = min(alpha, -tau / tmp_tau * 0.99999)
    end

    nbhd = Inf
    ls_tau = ls_kap = ls_tk = ls_mu = 0.0
    alphaprevok = true
    predfail = false
    nprediters = 0
    while true
        nprediters += 1

        # @. ls_tz = tz + alpha * tmp_tz
        # @. ls_ts = ts + alpha * tmp_ts
        # ls_tau = tau + alpha * tmp_tau
        # ls_kap = kap + alpha * tmp_kap
        # ls_tk = ls_tau * ls_kap
        # ls_mu = (dot(ls_ts, ls_tz) + ls_tk)/bnu

        step_in_direction(point, direction, alpha)

        # accept primal iterate if
        # - decreased alpha and it is the first inside the cone and beta-neighborhood or
        # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
        if ls_mu > 0.0 && ls_tau > 0.0 && ls_kap > 0.0 && Cones.incone(cone, ls_mu)
            # primal iterate is inside the cone
            # nbhd = Cones.calcnbhd!(g, ls_ts, ls_tz, ls_mu, cone) + abs2(ls_tk - ls_mu)
            nbhd = get_neighborhood(point2, solver)

            if nbhd < abs2(beta * ls_mu)
            # if nbhd < abs2(beta)
                # iterate is inside the beta-neighborhood
                if !alphaprevok || alpha > solver.predlsmulti
                    # either the previous iterate was outside the beta-neighborhood or increasing alpha again will make it > 1
                    if solver.predlinesearch
                        alphapred = alpha
                    end
                    break
                end

                alphaprevok = true
                alpha = alpha / solver.predlsmulti # increase alpha
                continue
            end
        end

        # primal iterate is either
        # - outside the cone or
        # - inside the cone and outside the beta-neighborhood and previous iterate was outside the beta-neighborhood
        if alpha < alphapredthres
            # alpha is very small, so predictor has failed
            predfail = true
            solver.verbose && println("predictor could not improve the solution ($nprediters line search steps); terminating")
            break
        end

        alphaprevok = false
        alpha = solver.predlsmulti * alpha # decrease alpha
    end
    if predfail
        solver.status = :PredictorFail
        break
    end


    # TODO two points, return the one we want
    return point
end

function get_correction_direction(point::HSDEPoint, direction::HSDEPoint, solver::HSDESolver)

    # TODO direction is another helper point (with preallocated space)

    # calculate correction direction
    @. tmp_tx = 0.0
    @. tmp_ty = 0.0
    for k in eachindex(cone.prmtvs)
        v1 = (cone.prmtvs[k].usedual ? ts : tz)
        @. @views tmp_tz[cone.idxs[k]] = -v1[cone.idxs[k]]
    end
    Cones.calcg!(g, cone)
    @. tmp_tz -= mu * g
    @. tmp_ts = 0.0

    (tmp_kap, tmp_tau) = LinearSystems.solvelinsys6!(tmp_tx, tmp_ty, tmp_tz, -kap + mu / tau, tmp_ts, 0.0, mu, tau, L)


    return direction
end

function perform_correction(point::HSDEPoint, point2::HSDEPoint, direction::HSDEPoint, solver::HSDESolver)

    # determine step length alpha by line search
    alpha = solver.alphacorr
    if tmp_kap < 0.0
        alpha = min(alpha, -kap / tmp_kap * 0.99999)
    end
    if tmp_tau < 0.0
        alpha = min(alpha, -tau / tmp_tau * 0.99999)
    end

    ncorrlsiters = 0
    while ncorrlsiters <= solver.maxcorrlsiters
        ncorrlsiters += 1

        # @. ls_tz = tz + alpha * tmp_tz
        # @. ls_ts = ts + alpha * tmp_ts
        # ls_tau = tau + alpha * tmp_tau
        # @assert ls_tau > 0.0
        # ls_kap = kap + alpha * tmp_kap
        # @assert ls_kap > 0.0
        # ls_mu = (dot(ls_ts, ls_tz) + ls_tau * ls_kap) / bnu

        step_in_direction(point, direction, alpha)


        if ls_mu > 0.0 && Cones.incone(cone, ls_mu)
            # primal iterate tx is inside the cone, so terminate line search
            break
        end

        # primal iterate tx is outside the cone
        if ncorrlsiters == solver.maxcorrlsiters
            # corrector failed
            corrfail = true
            solver.verbose && println("corrector could not improve the solution ($ncorrlsiters line search steps); terminating")
            break
        end

        alpha = solver.corrlsmulti * alpha # decrease alpha
    end
    if corrfail
        break
    end

    # TODO two points, return the one we want
    return point
end

function predict_then_correct(point::HSDEPoint, solver::HSDESolver)


    # prediction phase
    direction = get_prediction_direction(point, direction, solver)
    point = perform_prediction(point, point2, direction, solver)


    # skip correction phase if allowed and current iterate is in the eta-neighborhood
    if solver.corrcheck && nbhd <= abs2(eta * mu)
        continue
    end


    # correction phase
    ncorrsteps = 0
    while true # TODO maybe use for loop
        ncorrsteps += 1

        direction = get_correction_direction(point, direction, solver)
        point = perform_correction(point, point2, direction, solver)

        # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
        if ncorrsteps == solver.maxcorrsteps || solver.corrcheck
            # nbhd = Cones.calcnbhd!(g, ls_ts, ls_tz, mu, cone) + abs2(tau * kap - mu)
            nbhd = get_neighborhood(point, solver)

            @. ls_tz = tz
            @. ls_ts = ts
            if nbhd <= abs2(eta * mu)
                break
            elseif ncorrsteps == solver.maxcorrsteps
                # outside eta neighborhood, so corrector failed
                corrfail = true
                solver.verbose && println("corrector phase finished outside the eta-neighborhood ($ncorrsteps correction steps); terminating")
                break
            end
        end

        if corrfail
            solver.status = :CorrectorFail
            break
        end
    end

    return point
end
