#=
Copyright 2018, David Papp, Sercan Yildiz, and contributors

This code is an implementation of the algorithm for non-symmetric conic
optimization Alfonso, available at https://github.com/dpapp-github/alfonso
and analyzed in the paper:
D. Papp and S. Yildiz. On "A homogeneous interior-point algorithm for
nonsymmetric convex conic optimization". Available at
https://arxiv.org/abs/1712.00492.
=#

mutable struct AlfonsoModel <: MOI.AbstractModel
    # options
    verbose             # 0 if output is to be suppressed. 1 if progress is to be printed at each iteration. default value: 1.
    predlinesearch      # 0 if a fixed step size is to be used in the predictor step. 1 if the step size is to be determined via line search in the predictor step. default value: 1.
    maxcorrsteps        # maximum number of corrector steps. possible values: 1, 2, or 4. default value: 4.
    corrcheck           # 0 if maxcorrsteps corrector steps are to be performed at each corrector phase. 1 if the corrector phase can be terminated before maxcorrsteps corrector steps if the iterate is in the eta-neighborhood. default value: 1
    optimtol            # optimization tolerance parameter. default value: 1e-06.
    maxcorrlsiters      # maximum number of line search iterations in each corrector step. default value: 8.
    maxsmallpredsteps   # maximum number of predictor step size reductions allowed with respect to the safe fixed step size. default value: 8.
    maxitrefinesteps    # maximum number of iterative refinement steps in linear system solves. default value: 0.
    maxiter             # maximum number of iterations
    alphacorr           # corrector step size
    predlsmulti         # predictor line search step size multiplier
    corrlsmulti         # corrector line search step size multiplier
    itrefinethreshold   # iterative refinement success threshold
    beta                # large neighborhood parameter
    eta                 # small neighborhood parameter
    alphapredls         # initial predictor step size with line search
    alphapredfix        # fixed predictor step size
    alphapred           # initial predictor step size
    alphapredthreshold  # minimum predictor step size

    # data
    A               # constraint matrix
    b               # right-hand side vector
    c               # cost vector
    x0              # initial primal iterate
    gH::Function    # method for computing the gradient and Hessian of the barrier function
    gH_bnu          # complexity parameter of the augmented barrier (nu-bar)

    # output
    niterations     # total number of iterations
    alphapred       # predictor step size at each iteration
    betapred        # neighborhood parameter at the end of the predictor phase at each iteration
    etacorr         # neighborhood parameter at the end of the corrector phase at each iteration
    mu              # complementarity gap at each iteration
    x               # final value of the primal variables
    s               # final value of the dual slack variables
    y               # final value of the dual free variables
    tau             # final value of the tau-variable
    kappa           # final value of the kappa-variable
    pobj            # final primal objective value
    dobj            # final dual objective value
    dgap            # final duality gap
    cgap            # final complementarity gap
    rel_dgap        # final relative duality gap
    rel_cgap        # final relative complementarity gap
    pres            # final primal residuals
    dres            # final dual residuals
    pin             # final primal infeasibility
    din             # final dual infeasibility
    rel_pin         # final relative primal infeasibility
    rel_din         # final relative dual infeasibility
end


function MOI.optimize!(mod::AlfonsoModel)
    (A, b, c) = (mod.A, mod.b, mod.c)

    # verify data
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

    # build sparse lhs matrix
    lhs = [
        spzeros(m,m)  A             -b            spzeros(m,n)   spzeros(m,1)
        -A'           spzeros(n,n)  c             -speye(n)      spzeros(n,1)
        b'            -c'           0.0           spzeros(1,n)   -1.0
        spzeros(n,m)  speye(n)      spzeros(n,1)  speye(n)       spzeros(n,1)
        spzeros(1,m)  spzeros(1,n)  1.0           spzeros(1,n)   1.0
        ]

    # create arrays for storing iteration statistics
    alphapred = fill(NaN, mod.maxiter)
    betapred = fill(NaN, mod.maxiter)
    etacorr = fill(NaN, mod.maxiter)
    mu = fill(NaN, mod.maxiter)

    # calculate contants for termination criteria
    # TODO what norms are these
    term_pres = max(1.0, norm([A, b], Inf))
    term_dres = max(1.0, norm([A', speye(n), -c], Inf))
    term_comp = max(1.0, norm([-c', b', 1.0], Inf))

    # calculate the central primal-dual iterate corresponding to x0
    (soln_in, soln_g, soln_H, soln_L) = gH(x0)
    x = x0
    y = zeros(m)
    tau = 1.0
    s = -soln_g
    kappa = 1.0
    mu = (dot(x, s) + tau*kappa)/gH_bnu

    # main loop
    iter = 0
    while iter < mod.maxiter
        # check convergence criteria
        cx = dot(c, x)
        by = dot(b, y)

        metr_P = norm(A*x - tau*b, Inf)/term_pres
        metr_D = norm(A'*y + s - tau*c, Inf)/term_dres  # TODO blas for A'*y?
        metr_G = abs(cx - by + kappa)/term_comp
        metr_A = abs(cx - by)/(tau + abs(by))
        metr_O = cx/tau

        if (metr_P <= mod.optimtol) && (metr_D <= mod.optimtol)
            if metr_A <= mod.optimtol
                println("Problem is feasible and approximate optimal solution found; terminating")
                break
            elseif (metr_G <= mod.optimtol) && (tau <= mod.optimtol * 1e-02 * max(1.0, kappa))
                println("Problem is nearly primal or dual infeasible; terminating")
                break
            end
        elseif (tau <= mod.optimtol * 1e-02 * min(1.0, kappa)) && (mu <= mod.optimtol * 1e-02)
            println("Problem is ill-posed; terminating")
            break
        end

        # print progress metrics
        if mod.verbose
            @printf("%d: pobj=%d pIn=%d dIn=%d gap=%d tau=%d kap=%d mu=%d\n", iter, metr_O, metr_P, metr_D, metr_A, tau, kappa, mu)
        end

        iter += 1

        # predictor phase
        (alphapred, betapred, predfail) = predstep(mod)
        alphapred[iter] = alphapred
        betapred[iter] = betapred
        if predfail
            betapred[iter] = etacorr[iter-1]
            println("Predictor could not improve the solution; terminating")
            break
        end

        # corrector phase
        etacorr[iter] = betapred[iter]
        corrfail = false
        # skip if allowed and current iterate is in the eta-neighborhood
        if !mod.corrcheck || (etacorr[iter] > mod.eta)
            for corriter in 1:mod.maxcorrsteps
                # perform a correction step
                corrfail = corrstep(mod)
                if corrfail
                    println("Corrector could not improve the solution; terminating")
                    break
                end

                # finish if allowed and current iterate is in the eta-neighborhood, or if
                if (mod.corrcheck && (corriter < mod.maxcorrsteps)) || (mod == mod.maxcorrsteps)
                    etacorr[iter] = sqrt(sum(abs2, soln_L\soln_psi[1:end-1]) + (tau*soln_psi[end])^2)/mu
                    if etacorr[iter] <= mod.eta
                        break
                    end
                end
            end
            if etacorr[iter] > mod.eta
                println("Corrector phase finished outside the eta-neighborhood; terminating")
                break
            end
        end
        if corrfail
            break
        end

        mu[iter] = mu
    end

    println("Finished in $iter iterations")

    # calculate final solution and iteration statistics
    mod.niterations = iter

    mod.alphapred = alphapred[1:iter]
    mod.betapred = betapred[1:iter]
    mod.etacorr = etacorr[1:iter]
    mod.mu = mu[1:iter]

    mod.x = x./tau
    mod.s = s./tau
    mod.y = y./tau
    mod.tau = tau
    mod.kappa = kappa

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


# perform predictor step
function predstep(mod)
    (m, n) = size(mod.A)

    rhs = -vcat(mod.A*mod.x - mod.b*mod.tau, -mod.A'*mod.y + mod.c*mod.tau - mod.s, dot(mod.b, mod.y) - dot(mod.c, mod.x) - mod.kappa, mod.s, mod.kappa)
    (dy, dx, dtau, ds, dkappa) = computenewtondirection(mod, rhs)

    alpha = mod.alphapred
    betaalpha = Inf
    sa_innhd = 0
    alphaprevok = false
    predfail = false
    nsteps = 0
    
    while true
        nsteps += 1

        sa_y = mod.y + alpha*dy
        sa_x = mod.s + alpha*dx
        sa_tau = mod.tau + alpha*dtau
        sa_s = mod.s + alpha*ds
        sa_kappa = mod.kappa + alpha*dkappa

        (sa_in, sa_g, sa_H, sa_L) = gH(sa_x)

        if sa_in
            # primal iterate is inside the cone
            sa_mu = (dot(sa_x, sa_s) + sa_tau*sa_kappa)/mod.gH_bnu
            sa_psi = [sa_s; sa_kappa] + sa_mu*[sa_g; -1.0/sa_tau]
            betaalpha = sqrt(sum(abs2, sa_L\sa_psi(1:end-1)) + (sa_tau*sa_psi[end])^2)/sa_mu
            sa_innhd = (betaalpha < mod.beta)
        end

        if sa_in && sa_innhd
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

    return (alphapred, betapred, predfail)
end


# perform corrector step
function corrstep(mod)
    (m, n) = size(mod.A)

    rhs = vcat(zeros(m+n+1), -soln_psi)
    (dy, dx, dtau, ds, dkappa) = computenewtondirection(mod, rhs)

    alpha = mod.alphacorr
    corrfail = false

    # perform line search to make sure next primal iterate remains inside the cone
    for nsteps in 1:mod.maxcorrlsiters
        sa_y = mod.y + alpha*dy
        sa_x = mod.s + alpha*dx
        sa_tau = mod.tau + alpha*dtau
        sa_s = mod.s + alpha*ds
        sa_kappa = mod.kappa + alpha*dkappa

        (sa_in, sa_g, sa_H, sa_L) = gH(sa_x)

        # terminate line search if primal iterate is inside the cone
        if sa_in
            sa_mu = (dot(sa_x, sa_s) + sa_tau*sa_kappa)/mod.gH_bnu
            sa_psi = [sa_s; sa_kappa] + sa_mu*[sa_g; -1.0/sa_tau]
            break
        end

        alpha = mod.corrlsmulti*alpha
    end

    if !sa_in
        corrfail = true # corrector has failed
        solnalpha = soln
    end

    return corrfail
end


# set up the Newton system and compute Newton direction
# TODO optimize operations
function computenewtondirection(mod, L, rhs)
    (m, n) = size(mod.A)

    delta = solvesystem(mod, L, rhs)

    if mod.maxitrefinesteps > 0
        # checks to see if we need to refine the solution
        # TODO rcond and eps?
        if rcond(full(soln_H)) < mod.eps
            lhsnew = mod.lhs
            lhsnew(m+n+2:m+2n+1,m+1:m+n) = mu*soln_H
            lhsnew(end,m+n+1) = mod.mu/mod.tau^2

            res = residual3p(lhsnew, delta, rhs)
            resnorm = norm(res)

            for iter in 1:mod.maxitrefinesteps
                d = solvesystem(mod, L, rhs)
                deltanew = delta - d
                resnew = residual3p(lhsnew, deltanew, rhs)
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


# solve block system
# TODO optimize operations
function solvesystem(mod, L, rhs)
    (m, n) = size(mod.A)

    Hic = L'\(L\mod.c)
    HiAt = -L'\(L\mod.A')
    Hirxrs = L'\(L\(rhs[1:m] + rhs[m+n+2:m+2n+1]))

    LHSdydtau = [zeros(m), -mod.b; mod.b', mod.mu/mod.tau^2] - [mod.A; -mod.c']*[HiAt, Hic]/mod.mu
    rhsdydtau = [rhs[1:m]; (rhs[m+n+1] + rhs[end])] - [mod.A; -mod.c']*Hirxrs/mod.mu
    dydtau = LHSdydtau\rhsdydtau
    dx = (Hirxrs - [HiAt, Hic]*dydtau)/mod.mu

    return vcat(dydtau[1:m], dx, dydtau[m+1], (-rhs[1:m] - [mod.A', -mod.c]*dydtau), (-rhs[m+n+1] + dot(mod.b, dydtau[1:m]) - dot(mod.c, dx)))
end
