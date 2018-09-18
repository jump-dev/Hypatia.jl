
function solve_basic!(alf::AlfonsoOpt)
    starttime = time()

    (P, c, o, A, b, G, h) = (alf.P, alf.c, alf.o, alf.A, alf.b, alf.G, alf.h)
    (n, p, q) = (length(c), length(b), length(h))
    cone = alf.cone

    # preallocate arrays # TODO probably could get away with fewer. rename to temp_
    tx = similar(c)
    ts = similar(h)
    ty = similar(b)
    tz = similar(h)
    sa_ts = similar(ts)
    dir_ts = similar(ts)

    loadpnt!(cone, sa_ts)

    # calculate initial central primal-dual iterate (S5.3 of V.)
    # solve linear equation then step in interior direction of cone until inside cone
    # TODO use linsys solve
    alf.verbose && println("finding initial iterate")

    xyz = Symmetric([P A' G'; A zeros(p, p) zeros(p, q); G zeros(q, p) I]) \ [-c; b; h]
    tx .= xyz[1:n]
    ty .= xyz[n+1:n+p]
    sa_ts .= -xyz[n+p+1:n+p+q]
    ts .= sa_ts

    if !incone(cone)
        println("not in the cone")
        getintdir!(dir_ts, cone)
        alpha = 1.0 # TODO starting alpha maybe should depend on sa_ts (eg norm like in Alfonso) in case 1.0 is too large/small
        steps = 1
        while !incone(cone)
            sa_ts .= ts .+ alpha .* dir_ts
            alpha *= 1.2
            steps += 1
            if steps > 100
                error("cannot find initial iterate")
            end
        end
        @show alpha
        @show steps
        ts .= sa_ts
    end

    @assert incone(cone) # TODO delete
    calcg!(tz, cone)
    tz .*= -1.0

    tau = 1.0
    kap = 1.0
    mu = (dot(tz, ts) + tau*kap)/alf.bnu

    # TODO delete later
    @assert abs(1.0 - mu) < 1e-8
    @assert (tau*kap - mu)^2 + calcnbhd(mu, copy(tz), copy(tz), cone) < 1e-6

    alf.verbose && println("found initial iterate")

    # calculate prediction and correction step parameters
    (beta, eta, cpredfix) = getbetaeta(alf.maxcorrsteps, alf.bnu) # beta: large neighborhood parameter, eta: small neighborhood parameter
    alphapredfix = cpredfix/(eta + sqrt(2*eta^2 + alf.bnu)) # fixed predictor step size
    alphapredthres = (alf.predlsmulti^alf.maxpredsmallsteps)*alphapredfix # minimum predictor step size
    alphapredinit = (alf.predlinesearch ? min(100*alphapredfix, 0.9999) : alphapredfix) # predictor step size



    # (norm_c, norm_b, norm_h) = (norm(c), norm(b), norm(h))
    tol_pres = inv(1.0 + norm(b))
    tol_dres = inv(1.0 + norm(c))

    # main loop
    if alf.verbose
        println("Starting iteration")
        @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s\n", "iter", "p_obj", "d_obj", "gap", "p_inf", "d_inf", "mu")
        flush(stdout)
    end

    # "  Iter","prFeas","duFeas","muFeas","pobj","dobj","icertp","icertd","refine");ξ1()
    # Iter, rDu, rPr, rCp, pobj, dobj, p_infeas, d_infeas, rStep);ξ2()


    #     #=
    #     primal (over x):
    #      min  1/2 x'Px + c'x + o :
    #                   b - Ax == 0        (y)
    #                   h - Gx == s in K   (z)
    #     dual (over z,y,w):
    #      max  -1/2 w'Pw - b'y - h'z + o :
    #                   c + A'y + G'z == Pw   (x)
    #                               z in K*   (s)
    #
    #    optimality conditions are:
    #      c + Px + A'y + G'z == 0
    #      b - Ax             == 0
    #      h - Gx             == s
    #     and:
    #      z's == 0
    #        s in K
    #        z in K*
    #     =#


    alf.status = :StartedIterating
    alphapred = alphapredinit
    iter = 0
    while true
        # calculate convergence metrics
        



    #     ctx = dot(c, tx)
    #     bty = dot(b, ty)
    #     p_obj = ctx/tau
    #     d_obj = bty/tau
    #     rel_gap = abs(ctx - bty)/(tau + abs(bty))
    #     # p_res = -A*tx + b*tau
    #     mul!(p_res, A, tx)
    #     p_res .= tau .* b .- p_res
    #     p_inf = maximum(abs, p_res)/tol_pres
    #     # d_res = A'*ty - c*tau + ts
    #     mul!(d_res, A', ty)
    #     d_res .+= ts .- tau .* c
    #     d_inf = maximum(abs, d_res)/tol_dres
    #     abs_gap = -bty + ctx + kap
    #     compl = abs(abs_gap)/tol_compl
    #
    #     if alf.verbose
    #         # print iteration statistics
    #         @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter, p_obj, d_obj, rel_gap, p_inf, d_inf, tau, kap, mu)
    #         flush(stdout)
    #     end
    #
    #     # check convergence criteria
    #     if (p_inf <= alf.optimtol) && (d_inf <= alf.optimtol)
    #         if rel_gap <= alf.optimtol
    #             alf.verbose && println("Problem is feasible and an approximate optimal solution was found; terminating")
    #             alf.status = :Optimal
    #             break
    #         elseif (compl <= alf.optimtol) && (tau <= alf.optimtol*1e-02*max(1.0, kap))
    #             alf.verbose && println("Problem is nearly primal or dual infeasible; terminating")
    #             alf.status = :NearlyInfeasible
    #             break
    #         end
    #     elseif (tau <= alf.optimtol*1e-02*min(1.0, kap)) && (mu <= alf.optimtol*1e-02)
    #         alf.verbose && println("Problem is ill-posed; terminating")
    #         alf.status = :IllPosed
    #         break
    #     end
    #
    #     # check iteration limit
    #     iter += 1
    #     if iter >= alf.maxiter
    #         alf.verbose && println("Reached iteration limit; terminating")
    #         alf.status = :IterationLimit
    #         break
    #     end
    #
    #     # prediction phase
    #     # calculate prediction direction
    #     # x rhs is (ts - d_res), y rhs is p_res
    #     dir_tx .= ts .- d_res # temp for x rhs
    #     (y1, x1, y2, x2) = solvelinsys(y1, x1, y2, x2, mu, dir_tx, p_res, A, b, c, cone, HiAt, AHiAt)
    #
    #     dir_tau = (abs_gap - kap - dot(b, y2) + dot(c, x2))/(mu/tau^2 + dot(b, y1) - dot(c, x1))
    #     dir_ty .= y2 .+ dir_tau .* y1
    #     dir_tx .= x2 .+ dir_tau .* x1
    #     mul!(dir_ts, A', dir_ty)
    #     dir_ts .= dir_tau .* c .- dir_ts .- d_res
    #     dir_kap = -abs_gap + dot(b, dir_ty) - dot(c, dir_tx)
    #
    #     # determine step length alpha by line search
    #     alpha = alphapred
    #     nbhd = Inf
    #     alphaprevok = true
    #     predfail = false
    #     nprediters = 0
    #     while true
    #         nprediters += 1
    #
    #         sa_tx .= tx .+ alpha .* dir_tx
    #
    #         # accept primal iterate if
    #         # - decreased alpha and it is the first inside the cone and beta-neighborhood or
    #         # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
    #         if incone(cone)
    #             # primal iterate is inside the cone
    #             sa_ts .= ts .+ alpha .* dir_ts
    #             sa_tk = (tau + alpha*dir_tau)*(kap + alpha*dir_kap)
    #             sa_mu = (dot(sa_tx, sa_ts) + sa_tk)/alf.bnu
    #             nbhd = (sa_tk - mu)^2 + calcnbhd(sa_mu, sa_ts, g, cone)
    #
    #             if nbhd < abs2(beta*sa_mu)
    #                 # iterate is inside the beta-neighborhood
    #                 if !alphaprevok || (alpha > alf.predlsmulti)
    #                     # either the previous iterate was outside the beta-neighborhood or increasing alpha again will make it > 1
    #                     if alf.predlinesearch
    #                         alphapred = alpha
    #                     end
    #                     break
    #                 end
    #
    #                 alphaprevok = true
    #                 alpha = alpha/alf.predlsmulti # increase alpha
    #                 continue
    #             end
    #
    #             # # iterate is outside the beta-neighborhood
    #             # if alphaprevok # TODO technically this should be only if nprediters > 1, but it seems to work
    #             #     # previous iterate was inside the beta-neighborhood
    #             #     if alf.predlinesearch
    #             #         alphapred = alpha*alf.predlsmulti
    #             #     end
    #             #     break
    #             # end
    #         end
    #
    #         # primal iterate is either
    #         # - outside the cone or
    #         # - inside the cone and outside the beta-neighborhood and previous iterate was outside the beta-neighborhood
    #         if alpha < alphapredthres
    #             # alpha is very small, so predictor has failed
    #             predfail = true
    #             alf.verbose && println("Predictor could not improve the solution ($nprediters line search steps); terminating")
    #             alf.status = :PredictorFail
    #             break
    #         end
    #
    #         alphaprevok = false
    #         alpha = alf.predlsmulti*alpha # decrease alpha
    #     end
    #     # @show nprediters
    #     if predfail
    #         break
    #     end
    #
    #     # step distance alpha in the direction
    #     ty .+= alpha .* dir_ty
    #     tx .= sa_tx
    #     tau += alpha*dir_tau
    #     ts .+= alpha .* dir_ts
    #     kap += alpha*dir_kap
    #     mu = (dot(tx, ts) + tau*kap)/alf.bnu
    #
    #     # skip correction phase if allowed and current iterate is in the eta-neighborhood
    #     if alf.corrcheck && (nbhd < abs2(eta*mu))
    #         continue
    #     end
    #
    #     # correction phase
    #     corrfail = false
    #     ncorrsteps = 0
    #     while true
    #         ncorrsteps += 1
    #
    #         # calculate correction direction
    #         # x rhs is (ts + mu*g), y rhs is 0
    #         calcg!(g, cone)
    #         dir_tx .= ts .+ mu .* g # temp for x rhs
    #         dir_ty .= 0.0 # temp for y rhs
    #         (y1, x1, y2, x2) = solvelinsys(y1, x1, y2, x2, mu, dir_tx, dir_ty, A, b, c, cone, HiAt, AHiAt)
    #
    #         dir_tau = (mu/tau - kap - dot(b, y2) + dot(c, x2))/(mu/tau^2 + dot(b, y1) - dot(c, x1))
    #         dir_ty .= y2 .+ dir_tau .* y1
    #         dir_tx .= x2 .+ dir_tau .* x1
    #         # dir_ts = -A'*dir_ty + c*dir_tau
    #         mul!(dir_ts, A', dir_ty)
    #         dir_ts .= dir_tau .* c .- dir_ts
    #         dir_kap = dot(b, dir_ty) - dot(c, dir_tx)
    #
    #         # determine step length alpha by line search
    #         alpha = alf.alphacorr
    #         ncorrlsiters = 0
    #         while ncorrlsiters <= alf.maxcorrlsiters
    #             ncorrlsiters += 1
    #
    #             sa_tx .= tx .+ alpha .* dir_tx
    #
    #             if incone(cone)
    #                 # primal iterate tx is inside the cone, so terminate line search
    #                 break
    #             end
    #
    #             # primal iterate tx is outside the cone
    #             if ncorrlsiters == alf.maxcorrlsiters
    #                 # corrector failed
    #                 corrfail = true
    #                 alf.verbose && println("Corrector could not improve the solution ($ncorrlsiters line search steps); terminating")
    #                 alf.status = :CorrectorFail
    #                 break
    #             end
    #
    #             alpha = alf.corrlsmulti*alpha # decrease alpha
    #         end
    #         # @show ncorrlsiters
    #         if corrfail
    #             break
    #         end
    #
    #         # step distance alpha in the direction
    #         ty .+= alpha .* dir_ty
    #         tx .= sa_tx
    #         tau += alpha*dir_tau
    #         ts .+= alpha .* dir_ts
    #         kap += alpha*dir_kap
    #         mu = (dot(tx, ts) + tau*kap)/alf.bnu
    #
    #         # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
    #         if (ncorrsteps == alf.maxcorrsteps) || alf.corrcheck
    #             sa_ts .= ts
    #             if (tau*kap - mu)^2 + calcnbhd(mu, sa_ts, g, cone) <= abs2(eta*mu)
    #                 break
    #             elseif ncorrsteps == alf.maxcorrsteps
    #                 # nbhd_eta > eta, so corrector failed
    #                 corrfail = true
    #                 alf.verbose && println("Corrector phase finished outside the eta-neighborhood ($ncorrsteps correction steps); terminating")
    #                 alf.status = :CorrectorFail
    #                 break
    #             end
    #         end
    #     end
    #     # @show ncorrsteps
    #     if corrfail
    #         break
    #     end
    # end
    #
    # alf.verbose && println("\nFinished in $iter iterations\nInternal status is $(alf.status)\n")
    #
    # # calculate final solution and iteration statistics
    # alf.niters = iter
    #
    # tx ./= tau
    # alf.x = tx
    # ty ./= tau
    # alf.y = ty
    # alf.tau = tau
    # ts ./= tau
    # alf.s = ts
    # alf.kap = kap
    #
    # alf.pobj = dot(c, alf.x)
    # alf.dobj = dot(b, alf.y)
    # alf.dgap = alf.pobj - alf.dobj
    # alf.cgap = dot(alf.s, alf.x)
    # alf.rel_dgap = alf.dgap/(1.0 + abs(alf.pobj) + abs(alf.dobj))
    # alf.rel_cgap = alf.cgap/(1.0 + abs(alf.pobj) + abs(alf.dobj))
    #
    # # alf.pres = b - A*alf.x
    # mul!(p_res, A, alf.x)
    # p_res .= b .- p_res
    # alf.pres = p_res
    # # alf.dres = c - A'*alf.y - alf.s
    # mul!(d_res, A', alf.y)
    # d_res .= c .- d_res .- alf.s
    # alf.dres = d_res
    #
    # alf.pin = norm(alf.pres)
    # alf.din = norm(alf.dres)
    # alf.rel_pin = alf.pin/(1.0 + norm(b, Inf))
    # alf.rel_din = alf.din/(1.0 + norm(c, Inf))
    #
    # alf.solvetime = time() - starttime
#
#     return nothing
# end
#
# # TODO put this in cone.jl (used to be)
# function calcnbhd(mu, sa_tz, g, cone)
#     calcg!(g, cone)
#     sa_tz .+= mu .* g
#     calcHiarr!(g, sa_tz, cone)
#     return dot(sa_tz, g)
# end
