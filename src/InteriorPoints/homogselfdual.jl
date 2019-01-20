#=
Copyright 2018, Chris Coey and contributors

interior point type and functions for homogeneous self dual embedding algorithms
=#

mutable struct HSDEPoint <: InteriorPoint
    tx::Vector{Float64}
    ty::Vector{Float64}
    tz::Vector{Float64}
    ts::Vector{Float64}
    tau::Float64
    kap::Float64
end

get_mu(point::HSDEPoint, cone::Cones.Cone) = (dot(point.tz, point.ts) + point.tau * point.kap) / (1.0 + cone.barrier_param)

function step_in_direction(point::HSDEPoint, direction::HSDEPoint, alpha::Float64)
    @. point.tx += alpha * direction.tx
    @. point.ty += alpha * direction.ty
    @. point.tz += alpha * direction.tz
    @. point.ts += alpha * direction.ts
    point.tau += alpha * direction.tau
    point.kap += alpha * direction.kap
    return point
end


function get_initial_point!(point::HSDEPoint, model::HSDEModel)
    # TODO delete these if not so useful
    (c, A, b, G, h, cone) = (model.c, model.A, model.b, model.G, model.h, model.cone)
    (n, p, q) = (length(c), length(b), length(h))
    (tx, ty, tz, ts) = (point.tx, point.ty, point.tz, point.ts)

    point.tau = point.kap = 1.0

    Cones.get_central_point!(ts, tz, cone) # TODO scale like in alfonso?
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

function check_convergence(point::HSDEPoint, model::HSDEModel)
    # TODO delete these if not so useful
    (c, A, b, G, h, cone) = (model.c, model.A, model.b, model.G, model.h, model.cone)
    (tx, ty, tz, ts) = (point.tx, point.ty, point.tz, point.ts)

    # TODO rename tol variables to reflect which constraints they refer to
    # TODO consider 3 tolerances, one for primal equalities, one for primal cone constraint, and one for dual equalities


    # TODO calculate only once
    # calculate tolerances for convergence
    tol_res_tx = inv(max(1.0, norm(c)))
    tol_res_ty = inv(max(1.0, norm(b)))
    tol_res_tz = inv(max(1.0, norm(h)))


    # TODO preallocate
    # helper arrays for residuals, right-hand-sides, and search directions
    tmp_tx = similar(tx)
    tmp_tx2 = similar(tx)
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

    if hz + by < 0.0
        infres_pr = nres_x * tol_res_tx / (-hz - by)
    else
        infres_pr = NaN
    end
    if cx < 0.0
        infres_du = -max(nres_y * tol_res_ty, nres_z * tol_res_tz) / cx
    else
        infres_du = NaN
    end

    if model.verbose
        # print iteration statistics
        if iszero(model.iter)
            @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s %9s\n", "iter", "p_obj", "d_obj", "abs_gap", "rel_gap", "p_inf", "d_inf", "tau", "kap", "mu")
        end
        @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n",
            model.iter, obj_pr, obj_du, gap, relgap, nres_pr, nres_du, tau, kap, mu)
        flush(stdout)
    end

    # check convergence criteria
    # TODO nearly primal or dual infeasible or nearly optimal cases?
    if nres_pr <= model.tolfeas && nres_du <= model.tolfeas &&
        (gap <= model.tolabsopt || (!isnan(relgap) && relgap <= model.tolrelopt))
        model.verbose && println("optimal solution found; terminating")
        model.status = :Optimal
    elseif !isnan(infres_pr) && infres_pr <= model.tolfeas
        model.verbose && println("primal infeasibility detected; terminating")
        model.status = :PrimalInfeasible
    elseif !isnan(infres_du) && infres_du <= model.tolfeas
        model.verbose && println("dual infeasibility detected; terminating")
        model.status = :DualInfeasible
    elseif mu <= model.tolfeas * 1e-2 && tau <= model.tolfeas * 1e-2 * min(1.0, kap)
        model.verbose && println("ill-posedness detected; terminating")
        model.status = :IllPosed
    end

    return
end


function get_prediction_direction(point::HSDEPoint, direction::HSDEPoint, model::HSDEModel)

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

function perform_prediction(point::HSDEPoint, point2::HSDEPoint, direction::HSDEPoint, model::HSDEModel)

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

        @. ls_tz = tz + alpha * tmp_tz
        @. ls_ts = ts + alpha * tmp_ts
        ls_tau = tau + alpha * tmp_tau
        ls_kap = kap + alpha * tmp_kap
        ls_tk = ls_tau * ls_kap
        ls_mu = (dot(ls_ts, ls_tz) + ls_tk)/bnu

        # accept primal iterate if
        # - decreased alpha and it is the first inside the cone and beta-neighborhood or
        # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
        if ls_mu > 0.0 && ls_tau > 0.0 && ls_kap > 0.0 && Cones.incone(cone, ls_mu)
            # primal iterate is inside the cone
            nbhd = Cones.calcnbhd!(g, ls_ts, ls_tz, ls_mu, cone) + abs2(ls_tk - ls_mu)

            if nbhd < abs2(beta * ls_mu)
            # if nbhd < abs2(beta)
                # iterate is inside the beta-neighborhood
                if !alphaprevok || alpha > mdl.predlsmulti
                    # either the previous iterate was outside the beta-neighborhood or increasing alpha again will make it > 1
                    if mdl.predlinesearch
                        alphapred = alpha
                    end
                    break
                end

                alphaprevok = true
                alpha = alpha / mdl.predlsmulti # increase alpha
                continue
            end
        end

        # primal iterate is either
        # - outside the cone or
        # - inside the cone and outside the beta-neighborhood and previous iterate was outside the beta-neighborhood
        if alpha < alphapredthres
            # alpha is very small, so predictor has failed
            predfail = true
            mdl.verbose && println("predictor could not improve the solution ($nprediters line search steps); terminating")
            break
        end

        alphaprevok = false
        alpha = mdl.predlsmulti * alpha # decrease alpha
    end
    if predfail
        mdl.status = :PredictorFail
        break
    end


    # TODO two points, return the one we want
    return point
end

function get_correction_direction(point::HSDEPoint, direction::HSDEPoint, model::HSDEModel)

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

function perform_correction(point::HSDEPoint, model::HSDEModel)

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

    # determine step length alpha by line search
    alpha = mdl.alphacorr
    if tmp_kap < 0.0
        alpha = min(alpha, -kap / tmp_kap * 0.99999)
    end
    if tmp_tau < 0.0
        alpha = min(alpha, -tau / tmp_tau * 0.99999)
    end

    ncorrlsiters = 0
    while ncorrlsiters <= mdl.maxcorrlsiters
        ncorrlsiters += 1

        @. ls_tz = tz + alpha * tmp_tz
        @. ls_ts = ts + alpha * tmp_ts
        ls_tau = tau + alpha * tmp_tau
        @assert ls_tau > 0.0
        ls_kap = kap + alpha * tmp_kap
        @assert ls_kap > 0.0
        ls_mu = (dot(ls_ts, ls_tz) + ls_tau * ls_kap) / bnu

        if ls_mu > 0.0 && Cones.incone(cone, ls_mu)
            # primal iterate tx is inside the cone, so terminate line search
            break
        end

        # primal iterate tx is outside the cone
        if ncorrlsiters == mdl.maxcorrlsiters
            # corrector failed
            corrfail = true
            mdl.verbose && println("corrector could not improve the solution ($ncorrlsiters line search steps); terminating")
            break
        end

        alpha = mdl.corrlsmulti * alpha # decrease alpha
    end
    if corrfail
        break
    end

    # step distance alpha in the direction
    @. tx += alpha * tmp_tx
    @. ty += alpha * tmp_ty
    @. tz = ls_tz
    @. ts = ls_ts
    tau = ls_tau
    kap = ls_kap
    mu = ls_mu

    # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
    if ncorrsteps == mdl.maxcorrsteps || mdl.corrcheck
        nbhd = Cones.calcnbhd!(g, ls_ts, ls_tz, mu, cone) + abs2(tau * kap - mu)
        @. ls_tz = tz
        @. ls_ts = ts
        if nbhd <= abs2(eta * mu)
            break
        elseif ncorrsteps == mdl.maxcorrsteps
            # outside eta neighborhood, so corrector failed
            corrfail = true
            mdl.verbose && println("corrector phase finished outside the eta-neighborhood ($ncorrsteps correction steps); terminating")
            break
        end
    end


    return point
end




function predict_then_correct(point::HSDEPoint, model::HSDEModel)


    # prediction phase
    direction = get_prediction_direction(point, direction, model)
    point = perform_prediction(point, point2, direction, model)


    # skip correction phase if allowed and current iterate is in the eta-neighborhood
    if mdl.corrcheck && nbhd <= abs2(eta * mu)
        continue
    end


    # correction phase
    ncorrsteps = 0
    while true # TODO maybe use for loop
        ncorrsteps += 1

        direction = get_correction_direction(point, direction, model)
        point = perform_correction(point, point2, direction, model)

        # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
        if ncorrsteps == mdl.maxcorrsteps || mdl.corrcheck
            nbhd = Cones.calcnbhd!(g, ls_ts, ls_tz, mu, cone) + abs2(tau * kap - mu)
            @. ls_tz = tz
            @. ls_ts = ts
            if nbhd <= abs2(eta * mu)
                break
            elseif ncorrsteps == mdl.maxcorrsteps
                # outside eta neighborhood, so corrector failed
                corrfail = true
                mdl.verbose && println("corrector phase finished outside the eta-neighborhood ($ncorrsteps correction steps); terminating")
                break
            end
        end

        if corrfail
            mdl.status = :CorrectorFail
            break
        end
    end

    return point
end
