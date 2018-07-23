#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

an implementation of the algorithm for non-symmetric conic optimization Alfonso (https://github.com/dpapp-github/alfonso) and analyzed in the paper:
D. Papp and S. Yildiz. On "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization"
available at https://arxiv.org/abs/1712.00492
=#

function MOI.optimize!(opt::AlfonsoOptimizer)
    #=
    verify problem data, setup other algorithmic parameters and utilities
    =#
    dropzeros!(opt.A)
    (A, b, c) = (opt.A, opt.b, opt.c)
    (m, n) = size(A)

    if (m == 0) || (n == 0)
        error("input matrix A has trivial dimension $m x $n")
    end
    if m != length(b)
        error("dimension of vector b is $(length(b)), but number of rows in matrix A is $m")
    end
    if n != length(c)
        error("dimension of vector c is $(length(c)), but number of columns in matrix A is $n")
    end

    # # TODO check cones
    # cones = opt.cones
    #
    # # create function for computing the gradient and Hessian of the barrier function
    # function eval_gh(...)
    # end
    #
    # # calculate complexity parameter of the augmented barrier (nu-bar)
    # gh_bnu = NaN # TODO

    eval_gh = opt.eval_gh
    gh_bnu = opt.gh_bnu

    # set remaining algorithmic parameters based on precomputed safe values (from original authors)
    # parameters are chosen to make sure that each predictor step takes the current iterate from the eta-neighborhood to the beta-neighborhood and each corrector phase takes the current iterate from the beta-neighborhood to the eta-neighborhood. extra corrector steps are allowed to mitigate the effects of finite precision
    (beta, eta, cpredfix) = setbetaeta(opt.maxcorrsteps, gh_bnu) # beta: large neighborhood parameter, eta: small neighborhood parameter
    alphapredfix = cpredfix/(eta + sqrt(2*eta^2 + gh_bnu)) # fixed predictor step size
    alphapredls = min(100.0*alphapredfix, 0.9999) # initial predictor step size with line search
    alphapredthres = (opt.predlsmulti^opt.maxpredsmallsteps)*alphapredfix # minimum predictor step size
    alphapred = (opt.predlinesearch ? alphapredls : alphapredfix) # predictor step size

    #=
    setup data and functions needed in main loop
    =#
    # termination tolerances are infinity operator norms of submatrices of lhs
    tol_pres = max(1.0, maximum(sum(abs, A[i,:]) + abs(b[i]) for i in 1:m)) # first m rows
    tol_dres = max(1.0, maximum(sum(abs, A[:,j]) + abs(c[j]) + 1.0 for j in 1:n)) # next n rows
    tol_compl = max(1.0, maximum(abs, b), maximum(abs, c)) # row m+n+1

    # calculate initial primal iterate
    g = zeros(n)
    Hi = zeros(n,n)
    L = zeros(n,n)
    eval_gh(g, Hi, L, ones(n))
    # scaling factor for the primal problem
    rp = maximum((1.0 + abs(b[i]))/(1.0 + abs(sum(A[i,:]))) for i in 1:m)
    # scaling factor for the dual problem
    rd = maximum((1.0 + abs(g[j]))/(1.0 + abs(c[j])) for j in 1:n)
    # initial primal iterate
    tx0 = fill(sqrt(rp*rd), n)

    # calculate the central primal-dual iterate corresponding to the initial primal iterate
    incone = eval_gh(g, Hi, L, tx0)
    ty = zeros(m)
    tx = tx0
    tau = 1.0
    ts = -g
    kap = 1.0
    mu = (dot(tx, ts) + tau*kap)/gh_bnu

    # preallocate for test iterate and direction vectors
    sa_tx = similar(tx)
    sa_ts = similar(ts)
    dir_ty = similar(ty)
    dir_tx = similar(tx)
    dir_ts = similar(ts)

    #=
    main loop
    =#
    if opt.verbose
        @printf("\n%5s %11s %11s %9s %9s %9s %9s %9s %9s\n", "iter", "p_obj", "d_obj", "gap", "p_inf", "d_inf", "tau", "kap", "mu")
    end

    opt.status = :StartedIterating
    iter = 0
    while iter < opt.maxiter
        #=
        calculate convergence metrics, check criteria, print
        =#
        ctx = dot(c, tx)
        bty = dot(b, ty)
        p_obj = ctx/tau
        d_obj = bty/tau
        gap = abs(ctx - bty)/(tau + abs(bty))
        rhs_ty = -A*tx + b*tau
        p_inf = maximum(abs, rhs_ty)/tol_pres
        rhs_tx = A'*ty - c*tau + ts
        d_inf = maximum(abs, rhs_tx)/tol_dres
        rhs_tau = -bty + ctx + kap
        compl = abs(rhs_tau)/tol_compl

        if opt.verbose
            @printf("%5d %11.4e %11.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter, p_obj, d_obj, gap, p_inf, d_inf, tau, kap, mu)
            flush(stdout)
        end

        if (p_inf <= opt.optimtol) && (d_inf <= opt.optimtol)
            if gap <= opt.optimtol
                println("Problem is feasible and approximate optimal solution found; terminating")
                opt.status = :Optimal
                break
            elseif (compl <= opt.optimtol) && (tau <= opt.optimtol*1e-02*max(1.0, kap))
                println("Problem is nearly primal or dual infeasible; terminating")
                opt.status = :NearlyInfeasible
                break
            end
        elseif (tau <= opt.optimtol*1e-02*min(1.0, kap)) && (mu <= opt.optimtol*1e-02)
            println("Problem is ill-posed; terminating")
            opt.status = :IllPosed
            break
        end

        iter += 1

        #=
        prediction phase
        =#
        # determine prediction direction
        print("preddir  ")
        @time begin
            # TODO improve operations
            Hic = Hi*c
            HiAt = -Hi*A'
            Hirxrs = Hi*(rhs_tx - ts)

            lhsdydtau = [spzeros(m,m) -b; b' mu/tau^2] - ([A; -c']*[HiAt Hic])/mu
            rhsdydtau = [rhs_ty; (rhs_tau - kap)] - ([A; -c']*Hirxrs)/mu
            dydtau = lhsdydtau\rhsdydtau

            # @show typeof(Hic)
            # @show typeof(HiAt)
            # @show typeof(Hirxrs)
            # @show typeof(lhsdydtau)
            # @show typeof(rhsdydtau)
            # @show typeof(dydtau)

            dir_ty .= dydtau[1:m]
            dir_tx .= (Hirxrs - [HiAt Hic]*dydtau)/mu
            dir_tau = dydtau[m+1]
            dir_ts .= -rhs_tx - [A' -c]*dydtau
            dir_kap = -rhs_tau + dot(b, dydtau[1:m]) - dot(c, dir_tx)
        end

        # determine step length alpha by line search
        alpha = alphapred
        betaalpha = Inf
        alphaprevok = true
        alphaprev = 0.0
        betaalphaprev = Inf
        predfail = false
        nprediters = 0
        while true
            nprediters += 1

            sa_tx .= tx + alpha*dir_tx
            print(" eval_gh ")
            @time incone = eval_gh(g, Hi, L, sa_tx)
            if incone
                # primal iterate tx is inside the cone
                sa_ts .= ts + alpha*dir_ts
                sa_tk = (tau + alpha*dir_tau)*(kap + alpha*dir_kap)
                sa_mu = (dot(sa_tx, sa_ts) + sa_tk)/gh_bnu
                betaalpha = sqrt(sum(abs2, L\(sa_ts + sa_mu*g)) + (sa_tk - sa_mu)^2)/sa_mu

                if betaalpha < beta
                    # iterate is inside the beta-neighborhood
                    if !alphaprevok || (alpha > opt.predlsmulti)
                        # either the previous iterate was outside the beta-neighborhood or increasing alpha again will make it > 1
                        if opt.predlinesearch
                            alphapred = alpha
                        end
                        break
                    end

                    alphaprevok = true
                    alphaprev = alpha
                    betaalphaprev = betaalpha
                    alpha = alpha/opt.predlsmulti
                    continue
                end
            end

            # primal iterate tx is outside the cone and beta-neighborhood
            if alphaprevok && (nprediters > 1)
                # previous iterate was in the beta-neighborhood
                alpha = alphaprev
                betaalpha = betaalphaprev
                if opt.predlinesearch
                    alphapred = alpha
                end
                break
            end

            # the last two primal iterates were outside the beta-neighborhood
            if alpha < alphapredthres
                # alpha is very small, so predictor has failed
                predfail = true
                println("Predictor could not improve the solution; terminating")
                opt.status = :PredictorFail
                break
            end

            alphaprevok = false
            alphaprev = alpha
            betaalphaprev = betaalpha
            alpha = opt.predlsmulti*alpha
        end
        # @show nprediters
        if predfail
            break
        end

        # step distance alpha in the direction
        ty .+= alpha*dir_ty
        tx .+= alpha*dir_tx
        tau += alpha*dir_tau
        ts .+= alpha*dir_ts
        kap += alpha*dir_kap
        mu = (dot(tx, ts) + tau*kap)/gh_bnu
        print(" eval_gh ")
        @time eval_gh(g, Hi, L, tx)

        # skip correction phase if allowed and current iterate is in the eta-neighborhood
        if opt.corrcheck && (betaalpha <= eta)
            continue
        end

        #=
        correction phase: perform correction steps
        =#
        etacorr = betaalpha
        corrfail = false
        ncorrsteps = 0
        while true
            ncorrsteps += 1

            # calculate correction direction
            print("corrdir  ")
            @time begin
                # TODO improve operations
                Hic = Hi*c
                HiAt = -Hi*A'
                Hirxrs = Hi*(-ts - mu*g)

                lhsdydtau = [spzeros(m,m) -b; b' mu/tau^2] - ([A; -c']*[HiAt Hic])/mu
                rhsdydtau = [zeros(m); -kap + mu/tau] - ([A; -c']*Hirxrs)/mu
                dydtau = lhsdydtau\rhsdydtau

                dir_ty .= dydtau[1:m]
                dir_tx .= (Hirxrs - [HiAt Hic]*dydtau)/mu
                dir_tau = dydtau[m+1]
                dir_ts .= [-A' c]*dydtau
                dir_kap = dot(b, dydtau[1:m]) - dot(c, dir_tx)
            end

            # determine step length alpha by line search
            alpha = opt.alphacorr
            ncorrlsiters = 0
            while ncorrlsiters <= opt.maxcorrlsiters
                ncorrlsiters += 1

                sa_tx .= tx + alpha*dir_tx
                print(" eval_gh ")
                @time incone = eval_gh(g, Hi, L, sa_tx) # TODO only need incone until last iter
                if incone
                    # primal iterate tx is inside the cone, so terminate line search
                    break
                end

                # primal iterate tx is outside the cone
                if ncorrlsiters == opt.maxcorrlsiters
                    # corrector failed
                    corrfail = true
                    println("Corrector could not improve the solution; terminating")
                    opt.status = :CorrectorFail
                    break
                end

                alpha = opt.corrlsmulti*alpha
            end
            # @show ncorrlsiters
            if corrfail
                break
            end

            # step distance alpha in the direction
            ty .+= alpha*dir_ty
            tx .+= alpha*dir_tx
            tau += alpha*dir_tau
            ts .+= alpha*dir_ts
            kap += alpha*dir_kap
            mu = (dot(tx, ts) + tau*kap)/gh_bnu
            print(" eval_gh ")
            @time eval_gh(g, Hi, L, tx)

            # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
            if (ncorrsteps == opt.maxcorrsteps) || opt.corrcheck
                etacorr = sqrt(sum(abs2, L\(ts + mu*g)) + (tau*kap - mu)^2)/mu
                if etacorr <= eta
                    break
                elseif ncorrsteps == opt.maxcorrsteps
                    # etacorr > eta, so corrector failed
                    corrfail = true
                    println("Corrector phase finished outside the eta-neighborhood; terminating")
                    opt.status = :CorrectorFail
                    break
                end
            end
        end
        # @show ncorrsteps
        if corrfail
            break
        end
    end

    println("\nFinished in $iter iterations\nInternal status is $(opt.status)\n")

    #=
    calculate final solution and iteration statistics
    =#
    opt.niterations = iter

    opt.x = tx./tau
    opt.y = ty./tau
    opt.tau = tau
    opt.s = ts./tau
    opt.kap = kap

    opt.pobj = dot(c, opt.x)
    opt.dobj = dot(b, opt.y)
    opt.dgap = opt.pobj - opt.dobj
    opt.cgap = dot(opt.s, opt.x)
    opt.rel_dgap = opt.dgap/(1.0 + abs(opt.pobj) + abs(opt.dobj))
    opt.rel_cgap = opt.cgap/(1.0 + abs(opt.pobj) + abs(opt.dobj))

    opt.pres = b - A*opt.x
    opt.dres = c - A'*opt.y - opt.s
    opt.pin = norm(opt.pres)
    opt.din = norm(opt.dres)
    opt.rel_pin = opt.pin/(1.0 + norm(b, Inf))
    opt.rel_din = opt.din/(1.0 + norm(c, Inf))

    return nothing
end


function setbetaeta(maxcorrsteps, gh_bnu)
    if maxcorrsteps <= 2
        if gh_bnu < 10.0
            return (0.1810, 0.0733, 0.0225)
        elseif gh_bnu < 100.0
            return (0.2054, 0.0806, 0.0263)
        else
            return (0.2190, 0.0836, 0.0288)
        end
    elseif maxcorrsteps <= 4
        if gh_bnu < 10.0
            return (0.2084, 0.0502, 0.0328)
        elseif gh_bnu < 100.0
            return (0.2356, 0.0544, 0.0380)
        else
            return (0.2506, 0.0558, 0.0411)
        end
    else
        if gh_bnu < 10.0
            return (0.2387, 0.0305, 0.0429)
        elseif gh_bnu < 100.0
            return (0.2683, 0.0327, 0.0489)
        else
            return (0.2844, 0.0332, 0.0525)
        end
    end
end


#
# build sparse LHS matrix
# TODO this is not used by default, so ignore for now; only used if opt.maxitrefinesteps > 0
# lhs = [
#     spzeros(m,m)  A                 -b            spzeros(m,n)       spzeros(m,1)
#     -A'           spzeros(n,n)      c             sparse(-1.0I,n,n)  spzeros(n,1)
#     b'            -c'               0.0           spzeros(1,n)       -1.0
#     spzeros(n,m)  sparse(1.0I,n,n)  spzeros(n,1)  sparse(1.0I,n,n)   spzeros(n,1)
#     spzeros(1,m)  spzeros(1,n)      1.0           spzeros(1,n)       1.0
#     ]
# dropzeros!(lhs)
#
# create block solver function
# TODO optimize operations
# function solvesystem(rhs, L, mu, tau)
#     Hic = L'\(L\c)
#     HiAt = -L'\(L\A')
#     Hirxrs = L'\(L\(rhs[m+1:m+n] + rhs[m+n+2:m+2n+1]))
#
#     lhsdydtau = [zeros(m,m) -b; b' mu/tau^2] - ([A; -c']*[HiAt Hic])/mu
#     rhsdydtau = [rhs[1:m]; (rhs[m+n+1] + rhs[end])] - ([A; -c']*Hirxrs)/mu
#     dydtau = lhsdydtau\rhsdydtau
#     dx = (Hirxrs - [HiAt Hic]*dydtau)/mu
#
#     return (dydtau[1:m], dx, dydtau[m+1], (-rhs[m+1:m+n] - [A' -c]*dydtau), (-rhs[m+n+1] + dot(b, dydtau[1:m]) - dot(c, dx)))
# end
#
# # create Newton system solver function to compute Newton directions
# # TODO optimize operations
# function computenewtondirection(rhs, H, L, mu, tau)
#     delta = solvesystem(rhs, L, mu, tau)
#
#     if opt.maxitrefinesteps > 0
#         # TODO this is not used by default, so ignore for now
#         error("NOT IMPLEMENTED")
#         # checks to see if we need to refine the solution
#         # TODO rcond is not a function. and eps?
#         # if rcond(Array(H)) < eps() # TODO Base.LinAlg.LAPACK.gecon! # TODO really epsilon?\
#         #     lhsnew = copy(lhs)
#         #     lhsnew[m+n+2:m+2n+1,m+1:m+n] = mu*H
#         #     lhsnew[end,m+n+1] = mu/tau^2
#         #
#         #     # res = residual3p(lhsnew, delta, rhs)
#         #     res = lhsnew*delta - rhs # TODO needs to be in at least triple precision
#         #     resnorm = norm(res)
#         #
#         #     for refiter in 1:opt.maxitrefinesteps
#         #         d = solvesystem(rhs, L, mu, tau)
#         #         deltanew = delta - d
#         #
#         #         # res = residual3p(lhsnew, deltanew, rhs)
#         #         resnew = lhsnew*deltanew - rhs # TODO needs to be in at least triple precision
#         #         resnewnorm = norm(resnew)
#         #
#         #         # stop iterative refinement if there is not enough progress
#         #         if resnewnorm > opt.itrefinethres*resnorm
#         #             break
#         #         end
#         #
#         #         # update solution if residual norm is smaller
#         #         if resnewnorm < resnorm
#         #             delta = deltanew
#         #             res = resnew
#         #             resnorm = resnewnorm
#         #         end
#         #     end
#         # end
#     end
#
#     return (delta[1:m], delta[m+1:m+n], delta[m+n+1], delta[m+n+2:m+2n+1], delta[end])
# end
