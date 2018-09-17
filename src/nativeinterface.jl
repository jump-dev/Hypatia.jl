#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

an implementation of the algorithm for non-symmetric conic optimization Alfonso (https://github.com/dpapp-github/alfonso) and analyzed in the paper:
D. Papp and S. Yildiz. On "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization"
available at https://arxiv.org/abs/1712.00492
=#

# TODO add time limit option and use it in loop
mutable struct AlfonsoOpt
    # options
    verbose::Bool           # if true, prints progress at each iteration
    optimtol::Float64       # optimization tolerance parameter
    maxiter::Int            # maximum number of iterations
    predlinesearch::Bool    # if false, predictor step uses a fixed step size, else step size is determined via line search
    maxpredsmallsteps::Int  # maximum number of predictor step size reductions allowed with respect to the safe fixed step size
    predlsmulti::Float64    # predictor line search step size multiplier
    corrcheck::Bool         # if false, maxcorrsteps corrector steps are performed at each corrector phase, else the corrector phase can be terminated before maxcorrsteps corrector steps if the iterate is in the eta-neighborhood
    maxcorrsteps::Int       # maximum number of corrector steps (possible values: 1, 2, or 4)
    alphacorr::Float64      # corrector step size
    maxcorrlsiters::Int     # maximum number of line search iterations in each corrector step
    corrlsmulti::Float64    # corrector line search step size multiplier

    # problem data
    c::Vector{Float64}          # linear cost vector, size n
    o::Float64                  # objective offset scalar
    A::AbstractMatrix{Float64}  # equality constraint matrix, size p*n
    b::Vector{Float64}          # equality constraint vector, size p
    varK::Cone                  # primal variable cone object
    G::AbstractMatrix{Float64}  # cone constraint matrix, size q*n
    h::Vector{Float64}          # cone constraint vector, size q
    conK::Cone                  # primal constraint cone object

    # results
    status::Symbol          # solver status
    solvetime::Float64      # total solve time
    niters::Int             # total number of iterations
    y::Vector{Float64}      # final value of the dual free variables
    x::Vector{Float64}      # final value of the primal variables
    tau::Float64            # final value of the tau-variable
    s::Vector{Float64}      # final value of the dual slack variables
    kap::Float64            # final value of the kappa-variable
    pobj::Float64           # final primal objective value
    dobj::Float64           # final dual objective value
    dgap::Float64           # final duality gap
    cgap::Float64           # final complementarity gap
    rel_dgap::Float64       # final relative duality gap
    rel_cgap::Float64       # final relative complementarity gap
    pres::Vector{Float64}   # final primal residuals
    dres::Vector{Float64}   # final dual residuals
    pin::Float64            # final primal infeasibility
    din::Float64            # final dual infeasibility
    rel_pin::Float64        # final relative primal infeasibility
    rel_din::Float64        # final relative dual infeasibility

    # TODO match natural order of options listed above
    function AlfonsoOpt(verbose, optimtol, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, alphacorr, predlsmulti, corrlsmulti)
        alf = new()

        alf.verbose = verbose
        alf.optimtol = optimtol
        alf.maxiter = maxiter
        alf.predlinesearch = predlinesearch
        alf.maxpredsmallsteps = maxpredsmallsteps
        alf.maxcorrsteps = maxcorrsteps
        alf.corrcheck = corrcheck
        alf.maxcorrlsiters = maxcorrlsiters
        alf.alphacorr = alphacorr
        alf.predlsmulti = predlsmulti
        alf.corrlsmulti = corrlsmulti

        alf.status = :NotLoaded

        return alf
    end
end

function AlfonsoOpt(;
    verbose = false,
    optimtol = 1e-6,
    maxiter = 1e3,
    predlinesearch = true,
    maxpredsmallsteps = 8,
    maxcorrsteps = 8, # NOTE doubled in .m code
    corrcheck = true,
    maxcorrlsiters = 8,
    alphacorr = 1.0,
    predlsmulti = 0.7,
    corrlsmulti = 0.5,
    )

    if !(1e-10 <= optimtol <= 1e-2)
        error("optimtol must be from 1e-10 to 1e-2")
    end
    if maxiter < 1
        error("maxiter must be at least 1")
    end
    if maxpredsmallsteps < 1
        error("maxcorrsteps must be at least 1")
    end
    if !(1 <= maxcorrsteps <= 8)
        error("maxcorrsteps must be from 1 to 8")
    end

    return AlfonsoOpt(verbose, optimtol, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, alphacorr, predlsmulti, corrlsmulti)
end

get_status(alf::AlfonsoOpt) = alf.status
get_solvetime(alf::AlfonsoOpt) = alf.solvetime
get_niters(alf::AlfonsoOpt) = alf.niters
get_y(alf::AlfonsoOpt) = copy(alf.y)
get_x(alf::AlfonsoOpt) = copy(alf.x)
get_tau(alf::AlfonsoOpt) = alf.tau
get_s(alf::AlfonsoOpt) = copy(alf.s)
get_kappa(alf::AlfonsoOpt) = alf.kappa
get_pobj(alf::AlfonsoOpt) = alf.pobj
get_dobj(alf::AlfonsoOpt) = alf.dobj
get_dgap(alf::AlfonsoOpt) = alf.dgap
get_cgap(alf::AlfonsoOpt) = alf.cgap
get_rel_dgap(alf::AlfonsoOpt) = alf.rel_dgap
get_rel_cgap(alf::AlfonsoOpt) = alf.rel_cgap
get_pres(alf::AlfonsoOpt) = copy(alf.pres)
get_dres(alf::AlfonsoOpt) = copy(alf.dres)
get_pin(alf::AlfonsoOpt) = alf.pin
get_din(alf::AlfonsoOpt) = alf.din
get_rel_pin(alf::AlfonsoOpt) = alf.rel_pin
get_rel_din(alf::AlfonsoOpt) = alf.rel_din

# load and verify problem data, calculate algorithmic parameters
function load_data!(
    alf::AlfonsoOpt,
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    varK::Cone,
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    conK::Cone;
    check=true, # TODO later make false
    )

    # check data consistency
    if check
        n = length(c)
        p = length(b)
        q = length(h)
        @assert n > 0
        @assert p + q > 0
        if n != size(A, 2) || n != size(G, 2)
            error("number of variables is not consistent in A, G, and c")
        end
        if p != size(A, 1)
            error("number of constraint rows is not consistent in A and b")
        end
        if q != size(G, 1)
            error("number of constraint rows is not consistent in G and h")
        end
         # TODO do appropriate decomps at the same time, do preprocessing
        if rank(A) < p
            error("A matrix is not full-row-rank; some primal equalities may be redundant or inconsistent")
        end
        if rank(vcat(A, G)) < n
            error("[A' G'] is not full-row-rank; some dual equalities may be redundant (i.e. primal variables can be removed) or inconsistent")
        end
    end

    # save data in solver object
    alf.c = c
    alf.A = A
    alf.b = b
    alf.varK = varK
    alf.G = G
    alf.h = h
    alf.conK = conK
    alf.status = :Loaded

    return alf
end

# solve using basic method (no HSDE)
function solve_basic!(alf::AlfonsoOpt)
    starttime = time()

    (c, A, b, G, h) = (alf.c, alf.A, alf.b, alf.G, alf.h)
    (varK, conK) = (alf.varK, alf.conK)
    (n, p, q) = (length(c), length(b), length(h))
    bnu = barrierpar(varK) + barrierpar(conK) # complexity parameter of the barrier (sum of the primitive cone barrier parameters)

    # preallocate arrays
    tx = similar(c)
    ts = similar(h)
    ty = similar(b)
    tz = similar(ts)
    sa_ts = similar(ts)
    dir_ts = similar(ts)

    loadpnt!(cone, sa_ts)

    # calculate initial central primal-dual iterate (S5.3 of V.)
    # solve linear equation then step in interior direction of cone until inside cone
    alf.verbose && println("finding initial iterate")

    # TODO use linsys solve function
    xyz = Symmetric([diagm(getincidence!(tx, varK)) A' G'; A zeros(p, p) zeros(p, q); G zeros(q, p) I]) \ [-c; b; h]
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

    mu = dot(tz, ts)/bnu

    # TODO delete later
    @assert abs(1.0 - mu) < 1e-8
    @assert calcnbhd(tau*kap, mu, copy(tz), copy(tz), cone) < 1e-6

    alf.verbose && println("found initial iterate")


    # calculate prediction and correction step parameters
    # TODO put in prediction and correction step cache functions
    (beta, eta, cpredfix) = getbetaeta(alf.maxcorrsteps, bnu) # beta: large neighborhood parameter, eta: small neighborhood parameter
    alphapredfix = cpredfix/(eta + sqrt(2*eta^2 + bnu)) # fixed predictor step size
    alphapredthres = (alf.predlsmulti^alf.maxpredsmallsteps)*alphapredfix # minimum predictor step size
    alphapredinit = (alf.predlinesearch ? min(100*alphapredfix, 0.9999) : alphapredfix) # predictor step size


    # (norm_c, norm_b, norm_h) = (norm(c), norm(b), norm(h))
    # tol_pres = inv(1.0 + norm(b))
    # tol_dres = inv(1.0 + norm(c))

    # main loop
    if alf.verbose
        println("Starting iteration")
        @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s\n", "iter", "p_obj", "d_obj", "gap", "p_inf", "d_inf", "tau", "kap", "mu")
        flush(stdout)
    end

    alf.status = :StartedIterating
    alphapred = alphapredinit
    iter = 0
    while true
        # calculate residuals
        # TODO in-place
        res_tx = -A'*ty - G'*tz - c*tau
        res_ty = A*tx - b*tau
        res_tz = ts + G*x - h*tau
        res_tau = kap + dot(c, x) + dot(b, y) + dot(h, z)

        # TODO calculate convergence metrics
        # ctx = dot(c, tx)
        # bty = dot(b, ty)
        # p_obj = ctx/tau
        # d_obj = bty/tau
        # rel_gap = abs(ctx - bty)/(tau + abs(bty))
        # p_inf = maximum(abs, p_res)/alf.tol_pres
        # # d_res = A'*ty - c*tau + ts
        # mul!(d_res, A', ty)
        # d_res .+= ts .- tau .* c
        # d_inf = maximum(abs, d_res)/alf.tol_dres
        # abs_gap = -bty + ctx + kap
        # compl = abs(abs_gap)/alf.tol_compl
        #
        # if alf.verbose
        #     # print iteration statistics
        #     @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter, p_obj, d_obj, rel_gap, p_inf, d_inf, tau, kap, mu)
        #     flush(stdout)
        # end
        #
        # # check convergence criteria
        # if (p_inf <= alf.optimtol) && (d_inf <= alf.optimtol)
        #     if rel_gap <= alf.optimtol
        #         alf.verbose && println("Problem is feasible and an approximate optimal solution was found; terminating")
        #         alf.status = :Optimal
        #         break
        #     elseif (compl <= alf.optimtol) && (tau <= alf.optimtol*1e-02*max(1.0, kap))
        #         alf.verbose && println("Problem is nearly primal or dual infeasible; terminating")
        #         alf.status = :NearlyInfeasible
        #         break
        #     end
        # elseif (tau <= alf.optimtol*1e-02*min(1.0, kap)) && (mu <= alf.optimtol*1e-02)
        #     alf.verbose && println("Problem is ill-posed; terminating")
        #     alf.status = :IllPosed
        #     break
        # end

        # check iteration limit
        iter += 1
        if iter >= alf.maxiter
            alf.verbose && println("Reached iteration limit; terminating")
            alf.status = :IterationLimit
            break
        end

        # prediction phase
        # calculate prediction direction






        # x rhs is (ts - d_res), y rhs is p_res
        dir_tx .= ts .- d_res # temp for x rhs
        (y1, x1, y2, x2) = solvelinsys(y1, x1, y2, x2, mu, dir_tx, p_res, A, b, c, cone, HiAt, AHiAt)

        dir_tau = (abs_gap - kap - dot(b, y2) + dot(c, x2))/(mu/tau^2 + dot(b, y1) - dot(c, x1))
        dir_ty .= y2 .+ dir_tau .* y1
        dir_tx .= x2 .+ dir_tau .* x1
        mul!(dir_ts, A', dir_ty)
        dir_ts .= dir_tau .* c .- dir_ts .- d_res
        dir_kap = -abs_gap + dot(b, dir_ty) - dot(c, dir_tx)

        # determine step length alpha by line search
        alpha = alphapred
        nbhd = Inf
        alphaprevok = true
        predfail = false
        nprediters = 0
        while true
            nprediters += 1

            sa_tx .= tx .+ alpha .* dir_tx

            # accept primal iterate if
            # - decreased alpha and it is the first inside the cone and beta-neighborhood or
            # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
            if incone(cone)
                # primal iterate is inside the cone
                sa_ts .= ts .+ alpha .* dir_ts
                sa_tk = (tau + alpha*dir_tau)*(kap + alpha*dir_kap)
                sa_mu = (dot(sa_tx, sa_ts) + sa_tk)/bnu
                nbhd = calcnbhd(sa_tk, sa_mu, sa_ts, g, cone)

                if nbhd < abs2(alf.beta*sa_mu)
                    # iterate is inside the beta-neighborhood
                    if !alphaprevok || (alpha > alf.predlsmulti)
                        # either the previous iterate was outside the beta-neighborhood or increasing alpha again will make it > 1
                        if alf.predlinesearch
                            alphapred = alpha
                        end
                        break
                    end

                    alphaprevok = true
                    alpha = alpha/alf.predlsmulti # increase alpha
                    continue
                end

                # # iterate is outside the beta-neighborhood
                # if alphaprevok # TODO technically this should be only if nprediters > 1, but it seems to work
                #     # previous iterate was inside the beta-neighborhood
                #     if alf.predlinesearch
                #         alphapred = alpha*alf.predlsmulti
                #     end
                #     break
                # end
            end

            # primal iterate is either
            # - outside the cone or
            # - inside the cone and outside the beta-neighborhood and previous iterate was outside the beta-neighborhood
            if alpha < alf.alphapredthres
                # alpha is very small, so predictor has failed
                predfail = true
                alf.verbose && println("Predictor could not improve the solution ($nprediters line search steps); terminating")
                alf.status = :PredictorFail
                break
            end

            alphaprevok = false
            alpha = alf.predlsmulti*alpha # decrease alpha
        end
        # @show nprediters
        if predfail
            break
        end

        # step distance alpha in the direction
        ty .+= alpha .* dir_ty
        tx .= sa_tx
        tau += alpha*dir_tau
        ts .+= alpha .* dir_ts
        kap += alpha*dir_kap
        mu = (dot(tx, ts) + tau*kap)/bnu

        # skip correction phase if allowed and current iterate is in the eta-neighborhood
        if alf.corrcheck && (nbhd < abs2(alf.eta*mu))
            continue
        end

        # correction phase
        corrfail = false
        ncorrsteps = 0
        while true
            ncorrsteps += 1

            # calculate correction direction
            # x rhs is (ts + mu*g), y rhs is 0
            calcg!(g, cone)
            dir_tx .= ts .+ mu .* g # temp for x rhs
            dir_ty .= 0.0 # temp for y rhs
            (y1, x1, y2, x2) = solvelinsys(y1, x1, y2, x2, mu, dir_tx, dir_ty, A, b, c, cone, HiAt, AHiAt)

            dir_tau = (mu/tau - kap - dot(b, y2) + dot(c, x2))/(mu/tau^2 + dot(b, y1) - dot(c, x1))
            dir_ty .= y2 .+ dir_tau .* y1
            dir_tx .= x2 .+ dir_tau .* x1
            # dir_ts = -A'*dir_ty + c*dir_tau
            mul!(dir_ts, A', dir_ty)
            dir_ts .= dir_tau .* c .- dir_ts
            dir_kap = dot(b, dir_ty) - dot(c, dir_tx)

            # determine step length alpha by line search
            alpha = alf.alphacorr
            ncorrlsiters = 0
            while ncorrlsiters <= alf.maxcorrlsiters
                ncorrlsiters += 1

                sa_tx .= tx .+ alpha .* dir_tx

                if incone(cone)
                    # primal iterate tx is inside the cone, so terminate line search
                    break
                end

                # primal iterate tx is outside the cone
                if ncorrlsiters == alf.maxcorrlsiters
                    # corrector failed
                    corrfail = true
                    alf.verbose && println("Corrector could not improve the solution ($ncorrlsiters line search steps); terminating")
                    alf.status = :CorrectorFail
                    break
                end

                alpha = alf.corrlsmulti*alpha # decrease alpha
            end
            # @show ncorrlsiters
            if corrfail
                break
            end

            # step distance alpha in the direction
            ty .+= alpha .* dir_ty
            tx .= sa_tx
            tau += alpha*dir_tau
            ts .+= alpha .* dir_ts
            kap += alpha*dir_kap
            mu = (dot(tx, ts) + tau*kap)/bnu

            # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
            if (ncorrsteps == alf.maxcorrsteps) || alf.corrcheck
                sa_ts .= ts
                if calcnbhd(tau*kap, mu, sa_ts, g, cone) <= abs2(alf.eta*mu)
                    break
                elseif ncorrsteps == alf.maxcorrsteps
                    # nbhd_eta > eta, so corrector failed
                    corrfail = true
                    alf.verbose && println("Corrector phase finished outside the eta-neighborhood ($ncorrsteps correction steps); terminating")
                    alf.status = :CorrectorFail
                    break
                end
            end
        end
        # @show ncorrsteps
        if corrfail
            break
        end
    end

    alf.verbose && println("\nFinished in $iter iterations\nInternal status is $(alf.status)\n")

    # calculate final solution and iteration statistics
    alf.niters = iter

    tx ./= tau
    alf.x = tx
    ty ./= tau
    alf.y = ty
    alf.tau = tau
    ts ./= tau
    alf.s = ts
    alf.kap = kap

    alf.pobj = dot(c, alf.x)
    alf.dobj = dot(b, alf.y)
    alf.dgap = alf.pobj - alf.dobj
    alf.cgap = dot(alf.s, alf.x)
    alf.rel_dgap = alf.dgap/(1.0 + abs(alf.pobj) + abs(alf.dobj))
    alf.rel_cgap = alf.cgap/(1.0 + abs(alf.pobj) + abs(alf.dobj))

    # alf.pres = b - A*alf.x
    mul!(p_res, A, alf.x)
    p_res .= b .- p_res
    alf.pres = p_res
    # alf.dres = c - A'*alf.y - alf.s
    mul!(d_res, A', alf.y)
    d_res .= c .- d_res .- alf.s
    alf.dres = d_res

    alf.pin = norm(alf.pres)
    alf.din = norm(alf.dres)
    alf.rel_pin = alf.pin/(1.0 + norm(b, Inf))
    alf.rel_din = alf.din/(1.0 + norm(c, Inf))

    alf.solvetime = time() - starttime

    return nothing
end





# # solve using HSDE
# function solve_hsde!(alf::AlfonsoOpt)
#     starttime = time()
#
#     (c, A, b, G, h) = (alf.c, alf.A, alf.b, alf.G, alf.h)
#     (varK, conK) = (alf.varK, alf.conK)
#     (n, p, q) = (length(c), length(b), length(h))
#     bnu = 1.0 + barrierpar(varK) + barrierpar(conK) # complexity parameter nu-bar of the augmented barrier (sum of the primitive cone barrier parameters plus 1)
#
#     # preallocate arrays
#     tx = similar(c)
#     ts = similar(h)
#     ty = similar(b)
#     tz = similar(ts)
#     sa_ts = similar(ts)
#     dir_ts = similar(ts)
#
#     loadpnt!(cone, sa_ts)
#
#     # calculate initial central primal-dual iterate (S5.3 of V.)
#     # solve linear equation then step in interior direction of cone until inside cone
#     alf.verbose && println("finding initial iterate")
#
#     # TODO use linsys solve function
#     xyz = Symmetric([diagm(getincidence!(tx, varK)) A' G'; A zeros(p, p) zeros(p, q); G zeros(q, p) I]) \ [-c; b; h]
#     tx .= xyz[1:n]
#     ty .= xyz[n+1:n+p]
#     sa_ts .= -xyz[n+p+1:n+p+q]
#     ts .= sa_ts
#
#     if !incone(cone)
#         println("not in the cone")
#         getintdir!(dir_ts, cone)
#         alpha = 1.0 # TODO starting alpha maybe should depend on sa_ts (eg norm like in Alfonso) in case 1.0 is too large/small
#         steps = 1
#         while !incone(cone)
#             sa_ts .= ts .+ alpha .* dir_ts
#             alpha *= 1.2
#             steps += 1
#             if steps > 100
#                 error("cannot find initial iterate")
#             end
#         end
#         @show alpha
#         @show steps
#         ts .= sa_ts
#     end
#
#     @assert incone(cone) # TODO delete
#     calcg!(tz, cone)
#     tz .*= -1.0
#
#     tau = 1.0
#     kap = 1.0
#     mu = (dot(tz, ts) + tau*kap)/bnu
#
#     # TODO delete later
#     @assert abs(1.0 - mu) < 1e-8
#     @assert calcnbhd(tau*kap, mu, copy(tz), copy(tz), cone) < 1e-6
#
#     alf.verbose && println("found initial iterate")
#
#
#     # calculate prediction and correction step parameters
#     # TODO put in prediction and correction step cache functions
#     (beta, eta, cpredfix) = getbetaeta(alf.maxcorrsteps, bnu) # beta: large neighborhood parameter, eta: small neighborhood parameter
#     alphapredfix = cpredfix/(eta + sqrt(2*eta^2 + bnu)) # fixed predictor step size
#     alphapredthres = (alf.predlsmulti^alf.maxpredsmallsteps)*alphapredfix # minimum predictor step size
#     alphapredinit = (alf.predlinesearch ? min(100*alphapredfix, 0.9999) : alphapredfix) # predictor step size
#
#
#     # (norm_c, norm_b, norm_h) = (norm(c), norm(b), norm(h))
#     # tol_pres = inv(1.0 + norm(b))
#     # tol_dres = inv(1.0 + norm(c))
#
#     # main loop
#     if alf.verbose
#         println("Starting iteration")
#         @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s\n", "iter", "p_obj", "d_obj", "gap", "p_inf", "d_inf", "tau", "kap", "mu")
#         flush(stdout)
#     end
#
#     alf.status = :StartedIterating
#     alphapred = alphapredinit
#     iter = 0
#     while true
#         # calculate residuals
#         # TODO in-place
#         res_tx = -A'*ty - G'*tz - c*tau
#         res_ty = A*tx - b*tau
#         res_tz = ts + G*x - h*tau
#         res_tau = kap + dot(c, x) + dot(b, y) + dot(h, z)
#
#         # TODO calculate convergence metrics
#         # ctx = dot(c, tx)
#         # bty = dot(b, ty)
#         # p_obj = ctx/tau
#         # d_obj = bty/tau
#         # rel_gap = abs(ctx - bty)/(tau + abs(bty))
#         # p_inf = maximum(abs, p_res)/alf.tol_pres
#         # # d_res = A'*ty - c*tau + ts
#         # mul!(d_res, A', ty)
#         # d_res .+= ts .- tau .* c
#         # d_inf = maximum(abs, d_res)/alf.tol_dres
#         # abs_gap = -bty + ctx + kap
#         # compl = abs(abs_gap)/alf.tol_compl
#         #
#         # if alf.verbose
#         #     # print iteration statistics
#         #     @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter, p_obj, d_obj, rel_gap, p_inf, d_inf, tau, kap, mu)
#         #     flush(stdout)
#         # end
#         #
#         # # check convergence criteria
#         # if (p_inf <= alf.optimtol) && (d_inf <= alf.optimtol)
#         #     if rel_gap <= alf.optimtol
#         #         alf.verbose && println("Problem is feasible and an approximate optimal solution was found; terminating")
#         #         alf.status = :Optimal
#         #         break
#         #     elseif (compl <= alf.optimtol) && (tau <= alf.optimtol*1e-02*max(1.0, kap))
#         #         alf.verbose && println("Problem is nearly primal or dual infeasible; terminating")
#         #         alf.status = :NearlyInfeasible
#         #         break
#         #     end
#         # elseif (tau <= alf.optimtol*1e-02*min(1.0, kap)) && (mu <= alf.optimtol*1e-02)
#         #     alf.verbose && println("Problem is ill-posed; terminating")
#         #     alf.status = :IllPosed
#         #     break
#         # end
#
#         # check iteration limit
#         iter += 1
#         if iter >= alf.maxiter
#             alf.verbose && println("Reached iteration limit; terminating")
#             alf.status = :IterationLimit
#             break
#         end
#
#         # prediction phase
#         # calculate prediction direction
#
#
#
#
#
#
#         # x rhs is (ts - d_res), y rhs is p_res
#         dir_tx .= ts .- d_res # temp for x rhs
#         (y1, x1, y2, x2) = solvelinsys(y1, x1, y2, x2, mu, dir_tx, p_res, A, b, c, cone, HiAt, AHiAt)
#
#         dir_tau = (abs_gap - kap - dot(b, y2) + dot(c, x2))/(mu/tau^2 + dot(b, y1) - dot(c, x1))
#         dir_ty .= y2 .+ dir_tau .* y1
#         dir_tx .= x2 .+ dir_tau .* x1
#         mul!(dir_ts, A', dir_ty)
#         dir_ts .= dir_tau .* c .- dir_ts .- d_res
#         dir_kap = -abs_gap + dot(b, dir_ty) - dot(c, dir_tx)
#
#         # determine step length alpha by line search
#         alpha = alphapred
#         nbhd = Inf
#         alphaprevok = true
#         predfail = false
#         nprediters = 0
#         while true
#             nprediters += 1
#
#             sa_tx .= tx .+ alpha .* dir_tx
#
#             # accept primal iterate if
#             # - decreased alpha and it is the first inside the cone and beta-neighborhood or
#             # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
#             if incone(cone)
#                 # primal iterate is inside the cone
#                 sa_ts .= ts .+ alpha .* dir_ts
#                 sa_tk = (tau + alpha*dir_tau)*(kap + alpha*dir_kap)
#                 sa_mu = (dot(sa_tx, sa_ts) + sa_tk)/bnu
#                 nbhd = calcnbhd(sa_tk, sa_mu, sa_ts, g, cone)
#
#                 if nbhd < abs2(alf.beta*sa_mu)
#                     # iterate is inside the beta-neighborhood
#                     if !alphaprevok || (alpha > alf.predlsmulti)
#                         # either the previous iterate was outside the beta-neighborhood or increasing alpha again will make it > 1
#                         if alf.predlinesearch
#                             alphapred = alpha
#                         end
#                         break
#                     end
#
#                     alphaprevok = true
#                     alpha = alpha/alf.predlsmulti # increase alpha
#                     continue
#                 end
#
#                 # # iterate is outside the beta-neighborhood
#                 # if alphaprevok # TODO technically this should be only if nprediters > 1, but it seems to work
#                 #     # previous iterate was inside the beta-neighborhood
#                 #     if alf.predlinesearch
#                 #         alphapred = alpha*alf.predlsmulti
#                 #     end
#                 #     break
#                 # end
#             end
#
#             # primal iterate is either
#             # - outside the cone or
#             # - inside the cone and outside the beta-neighborhood and previous iterate was outside the beta-neighborhood
#             if alpha < alf.alphapredthres
#                 # alpha is very small, so predictor has failed
#                 predfail = true
#                 alf.verbose && println("Predictor could not improve the solution ($nprediters line search steps); terminating")
#                 alf.status = :PredictorFail
#                 break
#             end
#
#             alphaprevok = false
#             alpha = alf.predlsmulti*alpha # decrease alpha
#         end
#         # @show nprediters
#         if predfail
#             break
#         end
#
#         # step distance alpha in the direction
#         ty .+= alpha .* dir_ty
#         tx .= sa_tx
#         tau += alpha*dir_tau
#         ts .+= alpha .* dir_ts
#         kap += alpha*dir_kap
#         mu = (dot(tx, ts) + tau*kap)/bnu
#
#         # skip correction phase if allowed and current iterate is in the eta-neighborhood
#         if alf.corrcheck && (nbhd < abs2(alf.eta*mu))
#             continue
#         end
#
#         # correction phase
#         corrfail = false
#         ncorrsteps = 0
#         while true
#             ncorrsteps += 1
#
#             # calculate correction direction
#             # x rhs is (ts + mu*g), y rhs is 0
#             calcg!(g, cone)
#             dir_tx .= ts .+ mu .* g # temp for x rhs
#             dir_ty .= 0.0 # temp for y rhs
#             (y1, x1, y2, x2) = solvelinsys(y1, x1, y2, x2, mu, dir_tx, dir_ty, A, b, c, cone, HiAt, AHiAt)
#
#             dir_tau = (mu/tau - kap - dot(b, y2) + dot(c, x2))/(mu/tau^2 + dot(b, y1) - dot(c, x1))
#             dir_ty .= y2 .+ dir_tau .* y1
#             dir_tx .= x2 .+ dir_tau .* x1
#             # dir_ts = -A'*dir_ty + c*dir_tau
#             mul!(dir_ts, A', dir_ty)
#             dir_ts .= dir_tau .* c .- dir_ts
#             dir_kap = dot(b, dir_ty) - dot(c, dir_tx)
#
#             # determine step length alpha by line search
#             alpha = alf.alphacorr
#             ncorrlsiters = 0
#             while ncorrlsiters <= alf.maxcorrlsiters
#                 ncorrlsiters += 1
#
#                 sa_tx .= tx .+ alpha .* dir_tx
#
#                 if incone(cone)
#                     # primal iterate tx is inside the cone, so terminate line search
#                     break
#                 end
#
#                 # primal iterate tx is outside the cone
#                 if ncorrlsiters == alf.maxcorrlsiters
#                     # corrector failed
#                     corrfail = true
#                     alf.verbose && println("Corrector could not improve the solution ($ncorrlsiters line search steps); terminating")
#                     alf.status = :CorrectorFail
#                     break
#                 end
#
#                 alpha = alf.corrlsmulti*alpha # decrease alpha
#             end
#             # @show ncorrlsiters
#             if corrfail
#                 break
#             end
#
#             # step distance alpha in the direction
#             ty .+= alpha .* dir_ty
#             tx .= sa_tx
#             tau += alpha*dir_tau
#             ts .+= alpha .* dir_ts
#             kap += alpha*dir_kap
#             mu = (dot(tx, ts) + tau*kap)/bnu
#
#             # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
#             if (ncorrsteps == alf.maxcorrsteps) || alf.corrcheck
#                 sa_ts .= ts
#                 if calcnbhd(tau*kap, mu, sa_ts, g, cone) <= abs2(alf.eta*mu)
#                     break
#                 elseif ncorrsteps == alf.maxcorrsteps
#                     # nbhd_eta > eta, so corrector failed
#                     corrfail = true
#                     alf.verbose && println("Corrector phase finished outside the eta-neighborhood ($ncorrsteps correction steps); terminating")
#                     alf.status = :CorrectorFail
#                     break
#                 end
#             end
#         end
#         # @show ncorrsteps
#         if corrfail
#             break
#         end
#     end
#
#     alf.verbose && println("\nFinished in $iter iterations\nInternal status is $(alf.status)\n")
#
#     # calculate final solution and iteration statistics
#     alf.niters = iter
#
#     tx ./= tau
#     alf.x = tx
#     ty ./= tau
#     alf.y = ty
#     alf.tau = tau
#     ts ./= tau
#     alf.s = ts
#     alf.kap = kap
#
#     alf.pobj = dot(c, alf.x)
#     alf.dobj = dot(b, alf.y)
#     alf.dgap = alf.pobj - alf.dobj
#     alf.cgap = dot(alf.s, alf.x)
#     alf.rel_dgap = alf.dgap/(1.0 + abs(alf.pobj) + abs(alf.dobj))
#     alf.rel_cgap = alf.cgap/(1.0 + abs(alf.pobj) + abs(alf.dobj))
#
#     # alf.pres = b - A*alf.x
#     mul!(p_res, A, alf.x)
#     p_res .= b .- p_res
#     alf.pres = p_res
#     # alf.dres = c - A'*alf.y - alf.s
#     mul!(d_res, A', alf.y)
#     d_res .= c .- d_res .- alf.s
#     alf.dres = d_res
#
#     alf.pin = norm(alf.pres)
#     alf.din = norm(alf.dres)
#     alf.rel_pin = alf.pin/(1.0 + norm(b, Inf))
#     alf.rel_din = alf.din/(1.0 + norm(c, Inf))
#
#     alf.solvetime = time() - starttime
#
#     return nothing
# end

function calcnbhd(tk, mu, sa_ts, g, cone)
    calcg!(g, cone)
    sa_ts .+= mu .* g
    calcHiarr!(g, sa_ts, cone)
    return (tk - mu)^2 + dot(sa_ts, g)
end

function solvelinsys(y1, x1, y2, x2, mu, rhs_tx, rhs_ty, A, b, c, cone, HiAt, AHiAt)
    invmu = inv(mu)

    # TODO could ultimately be faster or more stable to do cholesky.L ldiv everywhere than to do full hessian ldiv
    calcHiarr!(HiAt, A', cone)
    HiAt .*= invmu
    mul!(AHiAt, A, HiAt)
    F = cholesky!(Symmetric(AHiAt))

    # TODO can parallelize 1 and 2
    # y2 = F\(rhs_ty + HiAt'*rhs_tx)
    mul!(y2, HiAt', rhs_tx)
    y2 .+= rhs_ty
    ldiv!(F, y2) # y2 done

    # x2 = Hi*invmu*(A'*y2 - rhs_tx)
    mul!(x2, A', y2)
    rhs_tx .= x2 .- rhs_tx # destroys rhs_tx
    rhs_tx .*= invmu
    calcHiarr!(x2, rhs_tx, cone) # x2 done

    # y1 = F\(b + HiAt'*c)
    mul!(y1, HiAt', c)
    y1 .+= b
    ldiv!(F, y1) # y1 done

    # x1 = Hi*invmu*(A'*y1 - c)
    mul!(rhs_tx, A', y1)
    rhs_tx .-= c
    rhs_tx .*= invmu
    calcHiarr!(x1, rhs_tx, cone) # x1 done

    return (y1, x1, y2, x2)
end

function getbetaeta(maxcorrsteps, bnu)
    if maxcorrsteps <= 2
        if bnu < 10.0
            return (0.1810, 0.0733, 0.0225)
        elseif bnu < 100.0
            return (0.2054, 0.0806, 0.0263)
        else
            return (0.2190, 0.0836, 0.0288)
        end
    elseif maxcorrsteps <= 4
        if bnu < 10.0
            return (0.2084, 0.0502, 0.0328)
        elseif bnu < 100.0
            return (0.2356, 0.0544, 0.0380)
        else
            return (0.2506, 0.0558, 0.0411)
        end
    else
        if bnu < 10.0
            return (0.2387, 0.0305, 0.0429)
        elseif bnu < 100.0
            return (0.2683, 0.0327, 0.0489)
        else
            return (0.2844, 0.0332, 0.0525)
        end
    end
end
