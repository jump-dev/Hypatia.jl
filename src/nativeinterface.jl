"""
solves a pair of primal and dual cone programs
    minimize    c'*x
    subject to  G*x + s = h
                A*x = b
                s in K
    maximize    -h'*z - b'*y
    subject to  G'*z + A'*y + c = 0
                z in K*
K is a convex cone defined as a Cartesian product of recognized primitive cones, and K* is its dual cone

the primal-dual optimality conditions are
    G*x + s = h,  A*x = b
    G'*z + A'*y + c = 0
    s in K, z in K*
    s'*z = 0

"""
mutable struct AlfonsoOpt
    # options
    verbose::Bool           # if true, prints progress at each iteration
    tolrelopt::Float64      # relative optimality gap tolerance
    tolabsopt::Float64      # absolute optimality gap tolerance
    tolfeas::Float64        # feasibility tolerance
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
    A::AbstractMatrix{Float64}  # equality constraint matrix, size p*n
    b::Vector{Float64}          # equality constraint vector, size p
    G::AbstractMatrix{Float64}  # cone constraint matrix, size q*n
    h::Vector{Float64}          # cone constraint vector, size q
    conK::Cone                  # primal constraint cone object

    Q1                       # QR factorization of A'
    Q2
    Ri

    # results
    status::Symbol          # solver status
    solvetime::Float64      # total solve time
    niters::Int             # total number of iterations

    x::Vector{Float64}      # final value of the primal variables
    s::Vector{Float64}      # final value of the primal cone variables
    y::Vector{Float64}      # final value of the dual free variables
    z::Vector{Float64}      # final value of the dual cone variables

    tau::Float64            # final value of the tau variable
    kap::Float64            # final value of the kappa variable
    mu::Float64             # final value of mu

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
    function AlfonsoOpt(verbose, tolrelopt, tolabsopt, tolfeas, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, alphacorr, predlsmulti, corrlsmulti)
        alf = new()

        alf.verbose = verbose
        alf.tolrelopt = tolrelopt
        alf.tolabsopt = tolabsopt
        alf.tolfeas = tolfeas
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
    tolrelopt = 1e-6,
    tolabsopt = 1e-7,
    tolfeas = 1e-7,
    maxiter = 5e2,
    predlinesearch = true,
    maxpredsmallsteps = 8,
    maxcorrsteps = 8,
    corrcheck = true,
    maxcorrlsiters = 8,
    alphacorr = 1.0,
    predlsmulti = 0.7,
    corrlsmulti = 0.5,
    )

    if min(tolrelopt, tolabsopt, tolfeas) < 1e-10 || max(tolrelopt, tolabsopt, tolfeas) > 1e-2
        error("tolrelopt, tolabsopt, tolfeas must be between 1e-10 and 1e-2")
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

    return AlfonsoOpt(verbose, tolrelopt, tolabsopt, tolfeas, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, alphacorr, predlsmulti, corrlsmulti)
end

get_status(alf::AlfonsoOpt) = alf.status
get_solvetime(alf::AlfonsoOpt) = alf.solvetime
get_niters(alf::AlfonsoOpt) = alf.niters

get_x(alf::AlfonsoOpt) = copy(alf.x)
get_s(alf::AlfonsoOpt) = copy(alf.s)
get_y(alf::AlfonsoOpt) = copy(alf.y)
get_z(alf::AlfonsoOpt) = copy(alf.z)

get_tau(alf::AlfonsoOpt) = alf.tau
get_kappa(alf::AlfonsoOpt) = alf.kappa
get_mu(alf::AlfonsoOpt) = alf.mu

get_pobj(alf::AlfonsoOpt) = dot(alf.c, alf.x)
get_dobj(alf::AlfonsoOpt) = -dot(alf.b, alf.y) - dot(alf.h, alf.z)


# load and verify problem data, calculate algorithmic parameters
function load_data!(
    alf::AlfonsoOpt,
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    conK::Cone;
    check=false, # check rank conditions
    )

    # check data consistency
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

    # perform QR decomposition of A' for use in linear system solves
    # TODO only use for factorization-based linear system solves
    # TODO reduce allocs, time
    # A' = [Q1 Q2] * [R1; 0]
    At_qr = issparse(A) ? qr(sparse(A')) : qr(A') # TODO this should be automatic in Julia for transpose

    Q1 = At_qr.Q[:,1:p]
    Q2 = At_qr.Q[:,p+1:n]
    # TODO problematic for sparse or large? versus:
    # Q = Array(At_qr.Q)
    # Q1 = Q[:,1:p]
    # Q2 = Q[:,p+1:n]

    R = At_qr.R # TODO check upper tri
    @assert istriu(R)
    Ri = inv(UpperTriangular(R))

    # TODO rank for qr decomp should be implemented in Julia - see https://github.com/JuliaLang/julia/blob/f8b52dab77415a22d28497f48407aca92fbbd4c3/stdlib/LinearAlgebra/src/qr.jl#L895
    if check
        # check rank conditions
        if issparse(A) || issparse(G)
            error("rank cannot currently be determined for sparse A or G")
        end
        if rank(A) < p # TODO change to rank(At_qr)
            error("A matrix is not full-row-rank; some primal equalities may be redundant or inconsistent")
        end
        if rank(vcat(A, G)) < n
            error("[A' G'] is not full-row-rank; some dual equalities may be redundant (i.e. primal variables can be removed) or inconsistent")
        end
    end

    alf.c = c
    alf.A = A
    alf.b = b
    alf.G = G
    alf.h = h
    alf.conK = conK
    alf.Q1 = Q1
    alf.Q2 = Q2
    alf.Ri = Ri
    alf.status = :Loaded

    return alf
end

# solve using homogeneous self-dual embedding
function solve!(alf::AlfonsoOpt)
    starttime = time()

    (c, A, b, G, h) = (alf.c, alf.A, alf.b, alf.G, alf.h)
    cone = alf.conK
    (n, p, q) = (length(c), length(b), length(h))
    bnu = 1.0 + barrierpar(cone) # complexity parameter nu-bar of the augmented barrier (sum of the primitive cone barrier parameters plus 1)

    # preallocate arrays
    tx = similar(c)
    ts = similar(h)
    ty = similar(b)
    tz = similar(ts)
    sa_ts = similar(ts)
    sa_tz = similar(tz)
    dir_ts = similar(ts)
    g = similar(ts)
    Hi = zeros(q, q)

    loadpnt!(cone, sa_ts)

    # calculate initial central primal-dual iterate (S5.3 of V.)
    # solve linear equation then step in interior direction of cone until inside cone
    alf.verbose && println("\nfinding initial iterate")

    # TODO use linsys solve function
    xyz = Symmetric([zeros(n, n) A' G'; A zeros(p, p) zeros(p, q); G zeros(q, p) I]) \ [-c; b; h] # TODO this is currently different from what CVXOPT does
    tx .= xyz[1:n]
    ty .= xyz[n+1:n+p]
    sa_ts .= -xyz[n+p+1:n+p+q]
    ts .= sa_ts

    if !incone(cone)
        getintdir!(dir_ts, cone)
        alpha = 1.0 # TODO starting alpha maybe should depend on sa_ts (eg norm like in Alfonso) in case 1.0 is too large/small
        steps = 1
        while !incone(cone)
            sa_ts .= ts .+ alpha .* dir_ts
            alpha *= 1.5
            steps += 1
            if steps > 25
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
    mu = (dot(tz, ts) + tau*kap)/bnu

    # TODO delete later
    @assert abs(1.0 - mu) < 1e-8
    @assert calcnbhd(tau*kap, mu, copy(tz), copy(tz), cone) < 1e-6

    alf.verbose && println("initial iterate found")

    # calculate tolerances for convergence
    tol_res_tx = inv(max(1.0, norm(c)))
    tol_res_ty = inv(max(1.0, norm(b)))
    tol_res_tz = inv(max(1.0, norm(h)))

    # calculate prediction and correction step parameters
    # TODO put in prediction and correction step cache functions
    (beta, eta, cpredfix) = getbetaeta(alf.maxcorrsteps, bnu) # beta: large neighborhood parameter, eta: small neighborhood parameter
    alphapredfix = cpredfix/(eta + sqrt(2*eta^2 + bnu)) # fixed predictor step size
    alphapredthres = (alf.predlsmulti^alf.maxpredsmallsteps)*alphapredfix # minimum predictor step size
    alphapredinit = (alf.predlinesearch ? min(100*alphapredfix, 0.9999) : alphapredfix) # predictor step size

    # main loop
    if alf.verbose
        println("starting iteration")
        @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s\n", "iter", "p_obj", "d_obj", "rel_gap", "p_inf", "d_inf", "tau", "kap", "mu")
        flush(stdout)
    end

    alf.status = :StartedIterating
    alphapred = alphapredinit
    iter = 0
    while true
        # calculate residuals
        # TODO in-place
        res_x = -A'*ty - G'*tz
        nres_x = norm(res_x)
        res_tx = res_x - c*tau
        nres_tx = norm(res_tx)/tau

        res_y = A*tx
        nres_y = norm(res_y)
        res_ty = res_y - b*tau
        nres_ty = norm(res_ty)/tau

        res_z = ts + G*tx
        nres_z = norm(res_z)
        res_tz = res_z - h*tau
        nres_tz = norm(res_tz)/tau

        (cx, by, hz) = (dot(c, tx), dot(b, ty), dot(h, tz))

        res_tau = kap + cx + by + hz

        obj_pr = cx/tau
        obj_du = -(by + hz)/tau

        gap = dot(tz, ts) # TODO is this right? maybe should adapt original alfonso conditions

        # TODO maybe add small epsilon to denominators that are zero to avoid NaNs, and get rid of isnans further down
        if obj_pr < 0.0
            relgap = gap/-obj_pr
        elseif obj_pr > 0.0
            relgap = gap/obj_du
        else
            relgap = NaN
        end

        nres_pr = max(nres_ty*tol_res_ty, nres_tz*tol_res_tz)
        nres_du = nres_tx*tol_res_tx

        if hz + by < 0.0
            infres_pr = nres_x*tol_res_tx/(-hz - by)
        else
            infres_pr = NaN
        end
        if cx < 0.0
            infres_du = -max(nres_y*tol_res_ty, nres_z*tol_res_tz)/cx
        else
            infres_du = NaN
        end

        if alf.verbose
            # print iteration statistics
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter, obj_pr, obj_du, relgap, nres_pr, nres_du, tau, kap, mu)
            flush(stdout)
        end

        # check convergence criteria
        # TODO nearly primal or dual infeasible or nearly optimal cases?
        if nres_pr <= alf.tolfeas && nres_du <= alf.tolfeas && (gap <= alf.tolabsopt || (!isnan(relgap) && relgap <= alf.tolrelopt))
            alf.verbose && println("optimal solution found; terminating")
            alf.status = :Optimal
            break
        elseif !isnan(infres_pr) && infres_pr <= alf.tolfeas
            alf.verbose && println("primal infeasibility detected; terminating")
            alf.status = :PrimalInfeasible
            break
        elseif !isnan(infres_du) && infres_du <= alf.tolfeas
            alf.verbose && println("dual infeasibility detected; terminating")
            alf.status = :DualInfeasible
            break
        elseif mu <= alf.tolfeas*1e-2 && tau <= alf.tolfeas*1e-2*min(1.0, kap)
            alf.verbose && println("ill-posedness detected; terminating")
            alf.status = :IllPosed
            break
        end

        # check iteration limit
        iter += 1
        if iter >= alf.maxiter
            alf.verbose && println("iteration limit reached; terminating")
            alf.status = :IterationLimit
            break
        end

        # prediction phase
        # calculate prediction direction
        sa_ts .= ts
        (dir_tx, dir_ty, dir_tz, dir_kap, dir_ts, dir_tau) = finddirection(alf, Hi, res_tx, res_ty, -tz, -kap, res_tz, res_tau, mu, tau)

        # determine step length alpha by line search
        alpha = alphapred
        nbhd = Inf
        alphaprevok = true
        predfail = false
        nprediters = 0
        while true
            nprediters += 1

            sa_ts .= ts .+ alpha .* dir_ts

            # accept primal iterate if
            # - decreased alpha and it is the first inside the cone and beta-neighborhood or
            # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
            if incone(cone)
                # primal iterate is inside the cone
                sa_tz .= tz .+ alpha .* dir_tz
                sa_tk = (tau + alpha*dir_tau)*(kap + alpha*dir_kap)
                sa_mu = (dot(sa_ts, sa_tz) + sa_tk)/bnu
                nbhd = calcnbhd(sa_tk, sa_mu, sa_tz, g, cone)

                if nbhd < abs2(beta*sa_mu)
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
            end

            # primal iterate is either
            # - outside the cone or
            # - inside the cone and outside the beta-neighborhood and previous iterate was outside the beta-neighborhood
            if alpha < alphapredthres
                # alpha is very small, so predictor has failed
                predfail = true
                alf.verbose && println("predictor could not improve the solution ($nprediters line search steps); terminating")
                alf.status = :PredictorFail
                break
            end

            alphaprevok = false
            alpha = alf.predlsmulti*alpha # decrease alpha
        end
        if predfail
            break
        end

        # step distance alpha in the direction
        tx .+= alpha .* dir_tx
        ty .+= alpha .* dir_ty
        tz .+= alpha .* dir_tz
        ts .= sa_ts
        tau += alpha*dir_tau
        kap += alpha*dir_kap
        mu = (dot(ts, tz) + tau*kap)/bnu

        # skip correction phase if allowed and current iterate is in the eta-neighborhood
        if alf.corrcheck && (nbhd < abs2(eta*mu))
            continue
        end

        # correction phase
        corrfail = false
        ncorrsteps = 0
        while true
            ncorrsteps += 1

            # calculate correction direction
            (dir_tx, dir_ty, dir_tz, dir_kap, dir_ts, dir_tau) = finddirection(alf, Hi, zeros(n), zeros(p), -(tz + mu*calcg!(g, cone)), -(kap - mu/tau), zeros(q), 0.0, mu, tau)

            # determine step length alpha by line search
            alpha = alf.alphacorr
            ncorrlsiters = 0
            while ncorrlsiters <= alf.maxcorrlsiters
                ncorrlsiters += 1

                sa_ts .= ts .+ alpha .* dir_ts
                if incone(cone)
                    # primal iterate tx is inside the cone, so terminate line search
                    break
                end

                # primal iterate tx is outside the cone
                if ncorrlsiters == alf.maxcorrlsiters
                    # corrector failed
                    corrfail = true
                    alf.verbose && println("corrector could not improve the solution ($ncorrlsiters line search steps); terminating")
                    alf.status = :CorrectorFail
                    break
                end

                alpha = alf.corrlsmulti*alpha # decrease alpha
            end
            if corrfail
                break
            end

            # step distance alpha in the direction
            tx .+= alpha .* dir_tx
            ty .+= alpha .* dir_ty
            tz .+= alpha .* dir_tz
            ts .= sa_ts
            tau += alpha*dir_tau
            kap += alpha*dir_kap
            mu = (dot(ts, tz) + tau*kap)/bnu

            # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
            if (ncorrsteps == alf.maxcorrsteps) || alf.corrcheck
                sa_tz .= tz
                nbhd = calcnbhd(tau*kap, mu, sa_tz, g, cone)
                # @show sqrt(nbhd)/mu
                if nbhd <= abs2(eta*mu)
                    break
                elseif ncorrsteps == alf.maxcorrsteps
                    # outside eta neighborhood, so corrector failed
                    corrfail = true
                    alf.verbose && println("corrector phase finished outside the eta-neighborhood ($ncorrsteps correction steps); terminating")
                    alf.status = :CorrectorFail
                    break
                end
            end
        end
        if corrfail
            break
        end
    end

    alf.verbose && println("\nfinished in $iter iterations; internal status is $(alf.status)\n")

    # calculate solution and iteration statistics
    alf.niters = iter
    alf.x = tx ./= tau
    alf.s = ts ./= tau
    alf.y = ty ./= tau
    alf.z = tz ./= tau
    alf.tau = tau
    alf.kap = kap
    alf.mu = mu
    alf.solvetime = time() - starttime

    return nothing
end

function calcnbhd(tk, mu, sa_tz, g, cone)
    calcg!(g, cone)
    sa_tz .+= mu .* g
    calcHiarr!(g, sa_tz, cone)
    return (tk - mu)^2 + dot(sa_tz, g)
end

function finddirection(alf, Hi, rhs_tx, rhs_ty, rhs_tz, rhs_kap, rhs_ts, rhs_tau, mu, tau)
    (c, A, b, G, h) = (alf.c, alf.A, alf.b, alf.G, alf.h)
    cone = alf.conK
    (n, p, q) = (length(c), length(b), length(h))
    (Q1, Q2, Ri) = (alf.Q1, alf.Q2, alf.Ri)

    # calcHiarr!(Hi, Matrix(1.0I, q, q), cone)

    g = zeros(q)
    calcg!(g, cone)
    H = Diagonal(g .* g) # TODO LP only

    # solve two symmetric systems and combine the solutions
    # use QR + cholesky method from CVXOPT
    # (1) eliminate equality constraints via QR of A'
    # (2) solve reduced system by cholesky

    # |0  A'  G'   | * |ux| = |bx|
    # |A  0   0    |   |uy|   |by|
    # |G  0  -Hi/mu|   |uz|   |bz|


    # A' = [Q1 Q2] * [R1; 0]
    # [Q1 Q2]' * G' * mu*H * G * [Q1 Q2]
    muH = Symmetric(mu*H)

    QH = G'*muH*G
    K22 = Q2'*QH*Q2 # TODO use syrk
    K11 = Q1'*QH*Q1
    K12 = Q1'*QH*Q2

    # factorize K22 = Q2' * G' * mu*H * G * Q2
    K22_ch = cholesky!(Symmetric(K22)) # TODO in-place; maybe bunch-kaufman

    # TODO refac systems 1 and 2
    # system 1
    (bx, by, muHbz) = (-c, b, muH*h)

    bxGHbz = bx + G'*muHbz
    Q1tx = Ri'*by
    Q2tx = K22_ch\(Q2'*bxGHbz - K12'*Q1tx)

    y1 = Ri*(Q1'*bxGHbz - K11*Q1tx - K12*Q2tx)
    x1 = Q1*Q1tx + Q2*Q2tx
    z1 = muH*G*x1 - muHbz

    # system 2
    (bx, by, muHbz) = (rhs_tx, -rhs_ty, -muH*rhs_ts - rhs_tz)

    bxGHbz = bx + G'*muHbz
    Q1tx = Ri'*by
    Q2tx = K22_ch\(Q2'*bxGHbz - K12'*Q1tx)

    y2 = Ri*(Q1'*bxGHbz - K11*Q1tx - K12*Q2tx)
    x2 = Q1*Q1tx + Q2*Q2tx
    z2 = muH*G*x2 - muHbz

    # combine
    dir_tau = ((rhs_tau + rhs_kap) + dot(c, x2) + dot(b, y2) + dot(h, z2))/(mu/tau/tau - dot(c, x1) - dot(b, y1) - dot(h, z1))
    dir_tx = x2 + dir_tau * x1
    dir_ty = y2 + dir_tau * y1
    dir_tz = z2 + dir_tau * z1
    dir_ts = -G*dir_tx + h*dir_tau - rhs_ts
    dir_kap = -dot(c, dir_tx) - dot(b, dir_ty) - dot(h, dir_tz) - rhs_tau

    return (dir_tx, dir_ty, dir_tz, dir_kap, dir_ts, dir_tau)
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
