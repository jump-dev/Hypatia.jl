#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

an implementation of the algorithm for non-symmetric conic optimization Alfonso (https://github.com/dpapp-github/alfonso) and analyzed in the paper:
D. Papp and S. Yildiz. On "A homogeneous interior-point algorithm for nonsymmetric convex conic optimization"
available at https://arxiv.org/abs/1712.00492
=#

function MOI.optimize!(opt::AlfonsoOptimizer)
    (A, b, c) = (opt.A, opt.b, opt.c)
    (m, n) = size(A)
    coneobjs = opt.cones
    coneidxs = opt.coneidxs

    # calculate complexity parameter of the augmented barrier (nu-bar)
    bnu = 1.0 + sum(barpar(ck) for ck in coneobjs) # TODO sum of the primitive cone barrier parameters (plus 1?)

    # create cone object functions related to primal cone barrier
    function load_tx(_tx::Vector{Float64})
        for k in eachindex(coneobjs)
            load_txk(coneobjs[k], _tx[coneidxs[k]])
        end
        return nothing
    end
    function check_incone()
        for k in eachindex(coneobjs)
            if !inconek(coneobjs[k])
                return false
            end
        end
        return true
    end
    function calc_g!(_g)
        for k in eachindex(coneobjs)
            _g[coneidxs[k]] .= calc_gk(coneobjs[k])
        end
        return _g
    end
    function calc_Hi_vec!(_Hi_vec, _v)
        for k in eachindex(coneobjs)
            _Hi_vec[coneidxs[k]] .= calc_Hik(coneobjs[k])*_v[coneidxs[k]]
        end
        return _Hi_vec
    end
    function calc_Hi_At()
        _Hi_At = spzeros(n,m)
        for k in eachindex(coneobjs)
            _Hi_At[coneidxs[k],:] .= calc_Hik(coneobjs[k])*A[:,coneidxs[k]] # TODO maybe faster with CSC to do IJV
        end
        return _Hi_At
    end
    function calc_nbhd(_ts, _mu, _tk)
        sumsqr = (_tk - _mu)^2
        for k in eachindex(coneobjs)
            sumsqr += sum(abs2, calc_Lk(coneobjs[k])\(_ts[coneidxs[k]] + _mu*calc_gk(coneobjs[k])))
        end
        return sqrt(sumsqr)/_mu
     end

    # set remaining algorithmic parameters based on precomputed safe values (from original authors)
    # parameters are chosen to make sure that each predictor step takes the current iterate from the eta-neighborhood to the beta-neighborhood and each corrector phase takes the current iterate from the beta-neighborhood to the eta-neighborhood. extra corrector steps are allowed to mitigate the effects of finite precision
    (beta, eta, cpredfix) = setbetaeta(opt.maxcorrsteps, bnu) # beta: large neighborhood parameter, eta: small neighborhood parameter
    alphapredfix = cpredfix/(eta + sqrt(2.0*eta^2 + bnu)) # fixed predictor step size
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

    # calculate initial primal iterate tx
    # scaling factor for the primal problem
    rp = maximum((1.0 + abs(b[i]))/(1.0 + abs(sum(A[i,:]))) for i in 1:m)
    # scaling factor for the dual problem
    g = ones(n)
    load_tx(g)
    @assert check_incone() # TODO will fail in general
    calc_g!(g)
    rd = maximum((1.0 + abs(g[j]))/(1.0 + abs(c[j])) for j in 1:n)
    # initial primal iterate
    tx = fill(sqrt(rp*rd), n)

    # calculate central primal-dual iterate
    load_tx(tx)
    @assert check_incone()
    ty = zeros(m)
    tau = 1.0
    ts = -calc_g!(g)
    kap = 1.0
    mu = (dot(tx, ts) + tau*kap)/bnu

    # preallocate for test iterate and direction vectors
    rhs_ty = similar(ty)
    rhs_tx = similar(tx)
    dir_ty = similar(ty)
    dir_tx = similar(tx)
    dir_ts = similar(ts)
    Hic = similar(tx)
    # HiAt = spzeros(n,m)
    Hirxrs = similar(tx)
    # lhsdydtau = spzeros(m+1,m+1)
    rhsdydtau = Vector{Float64}(undef,m+1)
    dydtau = similar(rhsdydtau)
    sa_tx = similar(tx)
    sa_ts = similar(ts)

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
        rhs_ty .= -A*tx + b*tau
        p_inf = maximum(abs, rhs_ty)/tol_pres
        rhs_tx .= A'*ty - c*tau + ts
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
        println("preddir  ")
        @time begin
            if iter > 1
                load_tx(tx)
            end
            invmu = 1.0/mu
            calc_Hi_vec!(Hic, c)
            Hic .*= invmu
            HiAt = calc_Hi_At()
            HiAt .*= invmu
            dir_ts .= invmu*(rhs_tx - ts)
            calc_Hi_vec!(Hirxrs, dir_ts)
        end
        @time begin
            lhsdydtau = [A*HiAt (-b - A*Hic); (b' - c'*HiAt) (mu/tau^2 + c'*Hic)]
            rhsdydtau .= [(rhs_ty - A*Hirxrs); (rhs_tau - kap + c'*Hirxrs)]
            dydtau .= lhsdydtau\rhsdydtau

            dir_ty .= dydtau[1:m]
            dir_tau = dydtau[m+1]
            dir_tx .= Hirxrs + HiAt*dir_ty - Hic*dir_tau
            dir_ts .= -rhs_tx - A'*dir_ty + c*dir_tau
            dir_kap = -rhs_tau + dot(b, dir_ty) - dot(c, dir_tx)
        end

        # determine step length alpha by line search
        alpha = alphapred
        nbhd_beta = Inf
        alphaprevok = true
        alphaprev = 0.0
        nbhd_betaprev = Inf
        predfail = false
        nprediters = 0
        while true
            nprediters += 1

            sa_tx .= tx + alpha*dir_tx
            print(" is_incone")
            load_tx(sa_tx)
            @time incone = check_incone()

            if incone
                # primal iterate tx is inside the cone
                sa_ts .= ts + alpha*dir_ts
                sa_tk = (tau + alpha*dir_tau)*(kap + alpha*dir_kap)
                sa_mu = (dot(sa_tx, sa_ts) + sa_tk)/bnu
                print(" calc_nbhd")
                @time nbhd_beta = calc_nbhd(sa_ts, sa_mu, sa_tk)
                # nbhd_beta = sqrt(sum(abs2, L\(sa_ts + sa_mu*g)) + (sa_tk - sa_mu)^2)/sa_mu

                if nbhd_beta < beta
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
                    nbhd_betaprev = nbhd_beta
                    alpha = alpha/opt.predlsmulti
                    continue
                end
            end

            # primal iterate tx is outside the cone and beta-neighborhood
            if alphaprevok && (nprediters > 1)
                # previous iterate was in the beta-neighborhood
                alpha = alphaprev
                nbhd_beta = nbhd_betaprev
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
            nbhd_betaprev = nbhd_beta
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
        mu = (dot(tx, ts) + tau*kap)/bnu

        # skip correction phase if allowed and current iterate is in the eta-neighborhood
        if opt.corrcheck && (nbhd_beta <= eta)
            continue
        end

        #=
        correction phase: perform correction steps
        =#
        nbhd_eta = nbhd_beta
        corrfail = false
        ncorrsteps = 0
        while true
            ncorrsteps += 1

            # calculate correction direction
            println("corrdir  ")
            @time begin
                # TODO use in-place BLAS?
                print(" calc_gHi ")
                load_tx(tx)
                @time Hi = calc_Hinv()
                @time g = calc_g!(g)

                Hic .= Hi*c/mu
                HiAt .= Hi*A'/mu
                Hirxrs .= Hi*(-ts/mu - g)

                lhsdydtau .= [A*HiAt (-b - A*Hic); (b' - c'*HiAt) (mu/tau^2 + c'*Hic)]
                rhsdydtau .= [-A*Hirxrs; (-kap + mu/tau + c'*Hirxrs)]
                dydtau .= lhsdydtau\rhsdydtau

                dir_ty .= dydtau[1:m]
                dir_tau = dydtau[m+1]
                dir_tx .= Hirxrs + HiAt*dir_ty - Hic*dir_tau
                dir_ts .= -A'*dir_ty + c*dir_tau
                dir_kap = dot(b, dir_ty) - dot(c, dir_tx)
            end

            # determine step length alpha by line search
            alpha = opt.alphacorr
            ncorrlsiters = 0
            while ncorrlsiters <= opt.maxcorrlsiters
                ncorrlsiters += 1

                sa_tx .= tx + alpha*dir_tx
                print(" is_incone")
                load_tx(sa_tx)
                @time incone = check_incone(sa_tx)

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
            tx .+= sa_tx
            tau += alpha*dir_tau
            ts .+= alpha*dir_ts
            kap += alpha*dir_kap
            mu = (dot(tx, ts) + tau*kap)/bnu

            # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
            if (ncorrsteps == opt.maxcorrsteps) || opt.corrcheck
                print(" calc_nbhd")
                @time nbhd_eta = calc_nbhd(ts, mu, tau*kap)
                # nbhd_eta = sqrt(sum(abs2, L\(ts + mu*g)) + (tau*kap - mu)^2)/mu

                if nbhd_eta <= eta
                    break
                elseif ncorrsteps == opt.maxcorrsteps
                    # nbhd_eta > eta, so corrector failed
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


function setbetaeta(maxcorrsteps, bnu)
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
