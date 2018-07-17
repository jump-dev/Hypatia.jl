#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

an implementation of the algorithm for non-symmetric conic optimization Alfonso (https://github.com/dpapp-github/alfonso) and analyzed in the paper:
D. Papp and S. Yildiz. On "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization"
available at https://arxiv.org/abs/1712.00492
=#

function MOI.optimize!(mod::AlfonsoOptimizer)
    #=
    verify problem data, setup other algorithmic parameters and utilities
    =#
    (A, b, c) = (mod.A, mod.b, mod.c)
    (m, n) = size(A)

    if m == 0 || n == 0
        error("input matrix A has trivial dimension $m x $n")
    end
    if m != length(b)
        error("dimension of vector b is $(length(b)), but number of rows in matrix A is $m")
    end
    if n != length(c)
        error("dimension of vector c is $(length(c)), but number of columns in matrix A is $n")
    end

    # TODO check cones
    cones = mod.cones

    # create function for computing the gradient and Hessian of the barrier function
    function eval_gh(tx)
        # TODO


        return (incone, g, H, L)
    end
    mod.eval_gh = eval_gh

    # calculate complexity parameter of the augmented barrier (nu-bar)
    mod.gh_bnu = NaN # TODO

    # set remaining algorithmic parameters based on precomputed safe values (from original authors):
    # parameters are chosen to make sure that each predictor step takes the current iterate from the eta-neighborhood to the beta-neighborhood and each corrector phase takes the current iterate from the beta-neighborhood to the eta-neighborhood. extra corrector steps are allowed to mitigate the effects of finite precision
    if mod.maxcorrsteps <= 2
        if mod.gh_bnu < 10.0
            mod.beta = 0.1810
            mod.eta = 0.0733
            cpredfix = 0.0225
        elseif mod.gh_bnu < 100.0
            mod.beta = 0.2054
            mod.eta = 0.0806
            cpredfix = 0.0263
        else
            mod.beta = 0.2190
            mod.eta = 0.0836
            cpredfix = 0.0288
        end
    elseif mod.maxcorrsteps <= 4
        if mod.gh_bnu < 10.0
            mod.beta = 0.2084
            mod.eta = 0.0502
            cpredfix = 0.0328
        elseif mod.gh_bnu < 100.0
            mod.beta = 0.2356
            mod.eta = 0.0544
            cpredfix = 0.0380
        else
            mod.beta = 0.2506
            mod.eta = 0.0558
            cpredfix = 0.0411
        end
    else
        if mod.gh_bnu < 10.0
            mod.beta = 0.2387
            mod.eta = 0.0305
            cpredfix = 0.0429
        elseif mod.gh_bnu < 100.0
            mod.beta = 0.2683
            mod.eta = 0.0327
            cpredfix = 0.0489
        else
            mod.beta = 0.2844
            mod.eta = 0.0332
            cpredfix = 0.0525
        end
    end

    mod.alphapredfix = cpredfix/(mod.eta + sqrt(2*mod.eta^2 + mod.gh_bnu))
    mod.alphapredls = min(100.0*mod.alphapredfix, 0.9999)
    mod.alphapredthreshold = (mod.predlsmulti^mod.maxpredsmallsteps)*mod.alphapredfix

    if !mod.predlinesearch
        # fixed predictor step size
        mod.alphapred = mod.alphapredfix
    else
        # initial predictor step size with line search
        mod.alphapred = mod.alphapredls
    end

    mod.status = :Loaded

    #=
    setup data and functions needed in main loop
    =#
    # build sparse LHS matrix
    lhs = [
        spzeros(m,m)  A             -b            spzeros(m,n)   spzeros(m,1)
        -A'           spzeros(n,n)  c             -speye(n)      spzeros(n,1)
        b'            -c'           0.0           spzeros(1,n)   -1.0
        spzeros(n,m)  speye(n)      spzeros(n,1)  speye(n)       spzeros(n,1)
        spzeros(1,m)  spzeros(1,n)  1.0           spzeros(1,n)   1.0
        ]

    # create block solver function
    # TODO optimize operations
    function solvesystem(rhs, L, mu, tau)
        Hic = L'\(L\c)
        HiAt = -L'\(L\A')
        Hirxrs = L'\(L\(rhs[1:m] + rhs[m+n+2:m+2n+1]))

        lhsdydtau = [zeros(m) -b; b' mu/tau^2] - [A; -c']*[HiAt, Hic]/mu
        rhsdydtau = [rhs[1:m]; (rhs[m+n+1] + rhs[end])] - [A; -c']*Hirxrs/mu
        dydtau = lhsdydtau\rhsdydtau
        dx = (Hirxrs - [HiAt, Hic]*dydtau)/mu

        return vcat(dydtau[1:m], dx, dydtau[m+1], (-rhs[1:m] - [A', -c]*dydtau), (-rhs[m+n+1] + dot(b, dydtau[1:m]) - dot(c, dx)))
    end

    # create Newton system solver function to compute Newton directions
    # TODO optimize operations
    function computenewtondirection(rhs, H, L, mu, tau)
        delta = solvesystem(rhs, L, mu, tau)

        if mod.maxitrefinesteps > 0
            # checks to see if we need to refine the solution
            # TODO rcond and eps?
            if rcond(full(H)) < mod.eps # TODO Base.LinAlg.LAPACK.gecon!
                lhsnew = copy(lhs)
                lhsnew[m+n+2:m+2n+1,m+1:m+n] = mu*H
                lhsnew[end,m+n+1] = mu/tau^2

                # res = residual3p(lhsnew, delta, rhs)
                res = lhsnew*delta - rhs # TODO needs to be in at least triple precision
                resnorm = norm(res)

                for iter in 1:mod.maxitrefinesteps
                    d = solvesystem(rhs, L, mu, tau)
                    deltanew = delta - d

                    # res = residual3p(lhsnew, deltanew, rhs)
                    resnew = lhsnew*deltanew - rhs # TODO needs to be in at least triple precision
                    resnewnorm = norm(resnew)

                    # stop iterative refinement if there is not enough progress
                    if resnewnorm > mod.itrefinethreshold*resnorm
                        break
                    end

                    # update solution if residual norm is smaller
                    if resnewnorm < resnorm
                        delta = deltanew
                        res = resnew
                        resnorm = resnewnorm
                    end
                end
            end
        end

        return (delta[1:m], delta[m+1:m+n], delta[m+n+1], delta[m+n+2:m+2n+1], delta[end])
    end

    # calculate contants for termination criteria
    # TODO what norms are these
    tol_pres = max(1.0, norm([A, b], Inf))
    tol_dres = max(1.0, norm([A', speye(n), -c], Inf))
    tol_compl = max(1.0, norm([-c', b', 1.0], Inf))

    # calculate initial primal iterate
    # scaling factor for the primal problem
    rp = maximum((1.0 + abs(b[i]))/(1.0 + abs(sum(A[i,:]))) for i in 1:m)
    # scaling factor for the dual problem
    g0 = mod.eval_gh(ones(n))[2]
    rd = maximum((1.0 + abs(g0[j]))/(1.0 + abs(c[j])) for j in 1:n)
    # initial primal iterate
    tx0 = fill(sqrt(rp*rd), n)

    # calculate the central primal-dual iterate corresponding to the initial primal iterate
    (incone, g, H, L) = mod.eval_gh(tx0)
    tx = tx0
    ty = zeros(m)
    tau = 1.0
    ts = -g
    kap = 1.0
    mu = (dot(tx, ts) + tau*kap)/mod.gh_bnu

    #=
    main loop
    =#
    all_alphapred = fill(NaN, mod.maxiter)
    all_betapred = fill(NaN, mod.maxiter)
    all_etacorr = fill(NaN, mod.maxiter)
    all_mu = fill(NaN, mod.maxiter)

    if mod.verbose
        @printf("\n%5s %10s %10s %7s %7s %7s %7s %7s %7s\n", "iter", "p_obj", "d_obj", "gap", "p_inf", "d_inf", "tau", "kap", "mu")
    end

    iter = 0
    mod.status = :StartedIterating
    while iter < mod.maxiter
        #=
        calculate convergence metrics, check criteria, print
        =#
        ctx = dot(c, tx)
        bty = dot(b, ty)
        p_obj = ctx/tau
        d_obj = bty/tau
        gap = abs(ctx - bty)/(tau + abs(bty))
        p_inf = norm(A*tx - tau*b, Inf)/tol_pres
        d_inf = norm(A'*ty + ts - tau*c, Inf)/tol_dres
        compl = abs(ctx - bty + kap)/tol_compl

        if mod.verbose
            @printf("%5d %10.6e %10.6e %7.3e %7.3e %7.3e %7.3e %7.3e %7.3e\n", iter, p_obj, d_obj, gap, p_inf, d_inf, tau, kap, mu)
        end

        if (p_inf <= mod.optimtol) && (d_inf <= mod.optimtol)
            if gap <= mod.optimtol
                println("Problem is feasible and approximate optimal solution found; terminating")
                mod.status = :Optimal
                break
            elseif (compl <= mod.optimtol) && (tau <= mod.optimtol*1e-02*max(1.0, kap))
                println("Problem is nearly primal or dual infeasible; terminating")
                mod.status = :NearlyInfeasible
                break
            end
        elseif (tau <= mod.optimtol*1e-02*min(1.0, kap)) && (mu <= mod.optimtol*1e-02)
            println("Problem is ill-posed; terminating")
            mod.status = :IllPosed
            break
        end

        iter += 1

        #=
        prediction phase: calculate prediction direction, perform line search
        =#
        rhs = -vcat(A*tx - b*tau, -A'*ty + c*tau - ts, dot(b, ty) - dot(c, tx) - kap, ts, kap)
        (dty, dtx, dtau, dts, dkap) = computenewtondirection(mod, rhs)

        alpha = mod.alphapred
        betaalpha = Inf
        sa_inconenhd = false
        alphaprevok = false
        predfail = false
        while true
            sa_ty = mod.ty + alpha*dty
            sa_tx = mod.tx + alpha*dtx
            sa_tau = mod.tau + alpha*dtau
            sa_ts = mod.ts + alpha*dts
            sa_kap = mod.kap + alpha*dkap

            (sa_incone, sa_g, sa_H, sa_L) = mod.eval_gh(sa_tx)

            if sa_incone
                # primal iterate is inside the cone
                sa_mu = (dot(sa_tx, sa_ts) + sa_tau*sa_kap)/mod.gh_bnu
                sa_psi = vcat(sa_ts + sa_mu*sa_g, sa_kap - sa_mu/sa_tau)
                betaalpha = sqrt(sum(abs2, sa_L\sa_psi[1:end-1]) + (sa_tau*sa_psi[end])^2)/sa_mu
                sa_inconenhd = (betaalpha < mod.beta)
            end

            if sa_incone && sa_inconenhd
                # iterate is inside the beta-neighborhood
                if !alphaprevok || (alpha > mod.predlsmulti)
                    # either the previous iterate was outside the beta-neighborhood or increasing alpha again will make it > 1
                    if mod.predlinesearch
                        mod.alphapred = alpha
                    end
                    break
                end
                alphaprevok = true
                alphaprev = alpha
                betaalphaprev = betaalpha
                solnalphaprev = solnalpha
                alpha = alpha/mod.predlsmulti
            else
                # iterate is outside the beta-neighborhood
                if alphaprevok
                    # previous iterate was in the beta-neighborhood
                    alpha = alphaprev
                    betaalpha = betaalphaprev
                    solnalpha = solnalphaprev
                    if mod.predlinesearch
                        mod.alphapred = alpha
                    end
                    break
                end

                if alpha < mod.alphapredthreshold
                    # last two iterates were outside the beta-neighborhood and alpha is very small, so predictor has failed
                    predfail = true # predictor has failed
                    alpha = 0.0
                    betaalpha = Inf
                    solnalpha = soln
                    if mod.predlinesearch
                        mod.alphapred = alpha
                    end
                    break
                end

                # alphaprev, betaalphaprev, solnalphaprev will not be used
                alphaprevok = false
                alphaprev = alpha
                betaalphaprev = betaalpha
                solnalphaprev = solnalpha
                alpha = mod.predlsmulti*alpha
            end
        end

        all_alphapred[iter] = alphapred
        all_betapred[iter] = betapred

        if predfail
            all_betapred[iter] = all_etacorr[iter-1]
            println("Predictor could not improve the solution; terminating")
            mod.status = :PredictorFail
            break
        end

        # skip correction phase if allowed and current iterate is in the eta-neighborhood
        all_etacorr[iter] = all_betapred[iter]
        if mod.corrcheck && (all_etacorr[iter] <= mod.eta)
            all_mu[iter] = mu
            continue
        end

        #=
        correction phase: perform correction steps
        =#
        corrfail = false
        for corriter in 1:mod.maxcorrsteps
            # calculate correction direction
            rhs = vcat(zeros(m+n+1), -psi)
            (dty, dtx, dtau, dts, dkap) = computenewtondirection(mod, rhs)

            # perform line search to ensure next primal iterate remains inside the cone
            alpha = mod.alphacorr
            for lsiter in 1:mod.maxcorrlsiters
                sa_ty = ty + alpha*dty
                sa_tx = tx + alpha*dtx
                sa_tau = tau + alpha*dtau
                sa_ts = ts + alpha*dts
                sa_kap = kap + alpha*dkap

                (sa_incone, sa_g, sa_H, sa_L) = mod.eval_gh(sa_tx)

                # terminate line search if primal iterate is inside the cone
                if sa_incone
                    sa_mu = (dot(sa_tx, sa_ts) + sa_tau*sa_kap)/mod.gh_bnu
                    sa_psi = vcat(sa_ts + sa_mu*sa_g, sa_kap - sa_mu/sa_tau)
                    break
                elseif lsiter == mod.maxcorrlsiters
                    println("Corrector could not improve the solution; terminating")
                    mod.status = :CorrectorFail
                    corrfail = true
                    break
                else
                    alpha = mod.corrlsmulti*alpha
                end
            end

            if corrfail
                break
            end

            # finish if allowed and current iterate is in the eta-neighborhood, or if
            if (mod.corrcheck && (corriter < mod.maxcorrsteps)) || (corriter == mod.maxcorrsteps)
                all_etacorr[iter] = sqrt(sum(abs2, L\psi[1:end-1]) + (tau*psi[end])^2)/mu
                if all_etacorr[iter] <= mod.eta
                    break
                end
            end
        end

        if corrfail
            break
        end

        if all_etacorr[iter] > mod.eta
            println("Corrector phase finished outside the eta-neighborhood; terminating")
            mod.status = :CorrectorFail
            break
        end

        all_mu[iter] = mu
    end

    println("Finished in $iter iterations\nInternal status is $status\n")

    #=
    calculate final solution and iteration statistics
    =#
    mod.niterations = iter
    mod.all_alphapred = all_alphapred[1:iter]
    mod.all_betapred = all_betapred[1:iter]
    mod.all_etacorr = all_etacorr[1:iter]
    mod.all_mu = all_mu[1:iter]

    mod.x = tx./tau
    mod.y = ty./tau
    mod.tau = tau
    mod.s = ts./tau
    mod.kap = kap

    mod.pobj = dot(c, mod.x)
    mod.dobj = dot(b, mod.y)
    mod.dgap = mod.pobj - mod.dobj
    mod.cgap = dot(mod.s, mod.x)
    mod.rel_dgap = mod.dgap/(1.0 + abs(mod.pobj) + abs(mod.dobj))
    mod.rel_cgap = mod.cgap/(1.0 + abs(mod.pobj) + abs(mod.dobj))

    mod.pres = b - A*mod.x
    mod.dres = c - A'*mod.y - mod.s
    mod.pin = norm(mod.pres)
    mod.din = norm(mod.dres)
    mod.rel_pin = mod.pin/(1.0 + norm(b, Inf))
    mod.rel_din = mod.din/(1.0 + norm(c, Inf))

    return nothing
end
