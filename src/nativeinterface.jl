#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz
=#

# model object containing options, problem data, linear system cache, and solution
mutable struct Optimizer
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
    cone::Cone                  # primal constraint cone object

    L::LinSysCache  # cache for linear system solves

    # results
    status::Symbol          # solver status
    solvetime::Float64      # total solve time
    niters::Int             # total number of iterations

    x::Vector{Float64}      # final value of the primal free variables
    s::Vector{Float64}      # final value of the primal cone variables
    y::Vector{Float64}      # final value of the dual free variables
    z::Vector{Float64}      # final value of the dual cone variables
    tau::Float64            # final value of the tau variable
    kap::Float64            # final value of the kappa variable
    mu::Float64             # final value of mu
    pobj::Float64           # final primal objective value
    dobj::Float64           # final dual objective value

    function Optimizer(verbose, tolrelopt, tolabsopt, tolfeas, maxiter, predlinesearch, maxpredsmallsteps, predlsmulti, corrcheck, maxcorrsteps, alphacorr, maxcorrlsiters, corrlsmulti)
        opt = new()

        opt.verbose = verbose
        opt.tolrelopt = tolrelopt
        opt.tolabsopt = tolabsopt
        opt.tolfeas = tolfeas
        opt.maxiter = maxiter
        opt.predlinesearch = predlinesearch
        opt.maxpredsmallsteps = maxpredsmallsteps
        opt.predlsmulti = predlsmulti
        opt.corrcheck = corrcheck
        opt.maxcorrsteps = maxcorrsteps
        opt.alphacorr = alphacorr
        opt.maxcorrlsiters = maxcorrlsiters
        opt.corrlsmulti = corrlsmulti

        opt.status = :NotLoaded

        return opt
    end
end

# initialize a model object
function Optimizer(;
    verbose = false,
    tolrelopt = 1e-6,
    tolabsopt = 1e-7,
    tolfeas = 1e-7,
    maxiter = 5e2,
    predlinesearch = true,
    maxpredsmallsteps = 15,
    predlsmulti = 0.7,
    corrcheck = true,
    maxcorrsteps = 15,
    alphacorr = 1.0,
    maxcorrlsiters = 15,
    corrlsmulti = 0.5,
    )

    if min(tolrelopt, tolabsopt, tolfeas) < 1e-10 || max(tolrelopt, tolabsopt, tolfeas) > 1e-2
        error("tolrelopt, tolabsopt, tolfeas must be between 1e-10 and 1e-2")
    end
    if maxiter < 1
        error("maxiter must be at least 1")
    end
    if maxpredsmallsteps < 1
        error("maxpredsmallsteps must be at least 1")
    end
    if maxcorrsteps < 1
        error("maxcorrsteps must be at least 1")
    end

    return Optimizer(verbose, tolrelopt, tolabsopt, tolfeas, maxiter, predlinesearch, maxpredsmallsteps, predlsmulti, corrcheck, maxcorrsteps, alphacorr, maxcorrlsiters, corrlsmulti)
end

get_status(opt::Optimizer) = opt.status
get_solvetime(opt::Optimizer) = opt.solvetime
get_niters(opt::Optimizer) = opt.niters

get_x(opt::Optimizer) = copy(opt.x)
get_s(opt::Optimizer) = copy(opt.s)
get_y(opt::Optimizer) = copy(opt.y)
get_z(opt::Optimizer) = copy(opt.z)

get_tau(opt::Optimizer) = opt.tau
get_kappa(opt::Optimizer) = opt.kappa
get_mu(opt::Optimizer) = opt.mu

get_pobj(opt::Optimizer) = dot(opt.c, opt.x)
get_dobj(opt::Optimizer) = -dot(opt.b, opt.y) - dot(opt.h, opt.z)




# check data for consistency
function check_data(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cone,
    )

    (n, p, q) = (length(c), length(b), length(h))
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

    @assert length(cone.prms) == length(cone.idxs)
    for k in eachindex(cone.prms)
        @assert dimension(cone.prms[k]) == length(cone.idxs[k])
    end

    return nothing
end


# preprocess data (optional)
function preprocess_data(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64};
    tol::Float64=1e-10, # presolve tolerance
    nsamp::Int=3, # number of samples in consistency check
    )

    (n, p, q) = (length(c), length(b), length(h))

    issparse(A) && error("not implemented for sparse A") # TODO fix


    # TODO delete
    if n < p
        println("number of equality constraints ($p) exceeds number of variables ($n)")
    end
    if rank(A) < p # TODO change to rank(F)
        println("A matrix is not full-row-rank; some primal equalities may be redundant or inconsistent")
    end
    if rank(vcat(A, G)) < n
        println("[A' G'] is not full-row-rank; some dual equalities may be redundant (i.e. primal variables can be removed) or inconsistent")
    end


    function helppreprocess(M, v)
        # NOTE different behavior of sparse and dense qr
        # TODO use a similar approach to the qr procedure in interpolation.jl
        # F = qr(M', Val(true))
        # keep_pnt = F.p[1:U]
        # pts = ipts[keep_pnt,:] # subset of points indexed with the support of w
        # P0 = M[keep_pnt,1:L] # subset of polynomial evaluations up to total degree d
        
        samp = rand(size(M, 2), nsamp)
        sol = M'\samp
        keepidxs = [sum(abs(sol[i,j]) for j in 1:nsamp) > tol for i in 1:size(M, 1)]
        isconsistent = norm(M*(M[keepidxs,:]\v[keepidxs,:]) - v, Inf) < tol

        return (isconsistent, keepidxs)
    end

    # preprocess primal equality constraints
    if p > 0
        (prconsistent, prkeepidxs) = helppreprocess(A, b)
        if !prconsistent
            error("some primal equality constraints are inconsistent")
        end
        A = A[prkeepidxs,:]
        b = b[prkeepidxs]
        # p = length(b)
    end

    # preprocess dual equality constraints
    (duconsistent, dukeepidxs) = helppreprocess(hcat(A', G'), c)
    if !duconsistent
        error("some dual equality constraints are inconsistent")
    end
    A = A[:,dukeepidxs]
    G = G[:,dukeepidxs]
    c = c[dukeepidxs]
    # n = length(c)

    # TODO note h doesn't change

    return (c, A, b, G, h)


    #
    #
    # # perform QR decomposition of A' for use in some linear system solvers
    # # TODO reduce allocs, improve efficiency
    # # A' = [Q1 Q2] * [R1; 0]
    # if issparse(A)
    #     # TODO don't reorder to match original: store indices and permute optimal solution
    #     alf.verbose && println("\nJulia is currently missing some sparse matrix methods that could improve performance; Hypatia may perform better if A is loaded as a dense matrix")
    #     # TODO currently using dense Q1, Q2, R - probably some should be sparse
    #     F = qr(sparse(A'))
    #     @assert length(F.prow) == n
    #     @assert length(F.pcol) == p
    #     @assert istriu(F.R)
    #
    #     Q = F.Q*Matrix(1.0I, n, n)
    #     Q1 = zeros(n, p)
    #     Q1[F.prow, F.pcol] = Q[:, 1:p]
    #     Q2 = zeros(n, n-p)
    #     Q2[F.prow, :] = Q[:, p+1:n]
    #     Ri = zeros(p, p)
    #     Ri[F.pcol, F.pcol] = inv(UpperTriangular(F.R))
    # else
    #     F = qr(A')
    #     @assert istriu(F.R)
    #
    #     Q = F.Q*Matrix(1.0I, n, n)
    #     Q1 = Q[:, 1:p]
    #     Q2 = Q[:, p+1:n]
    #     Ri = inv(UpperTriangular(F.R))
    #     @assert norm(A'*Ri - Q1) < 1e-8 # TODO delete later
    # end


    # return maps and new data
end





# verify problem data and load into model object
function load_data!(
    opt::Optimizer,
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cone;
    check::Bool=false, # check rank conditions
    linsyscache=QRCholCache, # linear system solver cache (see linsyssolvers folder)
    )

    opt.L = linsyscache(c, A, b, G, h)

    opt.c = c
    opt.A = A
    opt.b = b
    opt.G = G
    opt.h = h
    opt.cone = cone
    opt.status = :Loaded

    return opt
end

# solve using predictor-corrector algorithm based on homogeneous self-dual embedding
function solve!(opt::Optimizer)
    starttime = time()

    (c, A, b, G, h, cone) = (opt.c, opt.A, opt.b, opt.G, opt.h, opt.cone)
    (n, p, q) = (length(c), length(b), length(h))
    bnu = 1.0 + barrierpar(cone) # complexity parameter nu-bar of the augmented barrier (sum of the primitive cone barrier parameters plus 1)

    # preallocate arrays
    # primal and dual variables multiplied by tau
    tx = similar(c)
    ty = similar(b)
    tz = similar(h)
    ts = similar(h)
    # values during line searches
    ls_tz = similar(tz)
    ls_ts = similar(ts)
    # cone functions evaluate barrier derivatives at ls_ts
    loadpnt!(cone, ls_ts)
    # gradient evaluations at ls_ts of the barrier function for K
    g = similar(ts)
    H = Matrix(1.0I, q, q)
    # helper arrays for residuals, right-hand-sides, and search directions
    tmp_tx = similar(tx)
    tmp_tx2 = similar(tx)
    tmp_ty = similar(ty)
    tmp_tz = similar(tz)
    tmp_ts = similar(ts)

    # find initial primal-dual iterate
    opt.verbose && println("\nfinding initial iterate")

    # solve linear equation
    # TODO check equations
    # |0  A' G'| * |tx| = |0|
    # |A  0  0 |   |ty|   |b|
    # |G  0 -I |   |ts|   |h|
    @. tx = 0.0
    @. ty = -b
    @. tz = -h
    solvelinsys3!(tx, ty, tz, H, opt.L)
    @. ts = -tz
    @. ls_ts = ts

    # from ts, step along interior direction of cone until ts is inside cone
    if !incone(cone)
        getintdir!(tmp_ts, cone)
        alpha = 1.0 # TODO starting alpha maybe should depend on ls_ts (eg norm like in Hypatia) in case 1.0 is too large/small
        steps = 1
        @. ls_ts += alpha*tmp_ts
        while !incone(cone)
            steps += 1
            if steps > 25
                error("cannot find initial iterate")
            end
            alpha *= 1.5
            @. ls_ts = ts + alpha*tmp_ts
        end
        opt.verbose && println("$steps steps taken for initial iterate")
        @. ts = ls_ts
    end

    calcg!(tz, cone)
    @. tz *= -1.0
    tau = 1.0
    kap = 1.0
    mu = (dot(tz, ts) + tau*kap)/bnu
    @assert !isnan(mu)
    @assert abs(1.0 - mu) < 1e-10
    # @assert calcnbhd(tau*kap, mu, copy(tz), copy(tz), cone) < 1e-6

    opt.verbose && println("initial iterate found")

    # calculate tolerances for convergence
    tol_res_tx = inv(max(1.0, norm(c)))
    tol_res_ty = inv(max(1.0, norm(b)))
    tol_res_tz = inv(max(1.0, norm(h)))

    # calculate prediction and correction step parameters
    (beta, eta, cpredfix) = getbetaeta(opt.maxcorrsteps, bnu) # beta: large neighborhood parameter, eta: small neighborhood parameter
    alphapredfix = cpredfix/(eta + sqrt(2*eta^2 + bnu)) # fixed predictor step size
    alphapredthres = (opt.predlsmulti^opt.maxpredsmallsteps)*alphapredfix # minimum predictor step size
    alphapredinit = (opt.predlinesearch ? min(100*alphapredfix, 0.9999) : alphapredfix) # predictor step size

    # main loop
    if opt.verbose
        println("starting iteration")
        @printf("\n%5s %12s %12s %9s %9s %9s %9s %9s %9s\n", "iter", "p_obj", "d_obj", "rel_gap", "p_inf", "d_inf", "tau", "kap", "mu")
        flush(stdout)
    end

    opt.status = :StartedIterating
    alphapred = alphapredinit
    iter = 0
    while true
        # calculate residuals and convergence parameters
        invtau = inv(tau)

        # tmp_tx = -A'*ty - G'*tz - c*tau
        mul!(tmp_tx2, A', ty)
        mul!(tmp_tx, G', tz)
        @. tmp_tx = -tmp_tx2 - tmp_tx
        nres_x = norm(tmp_tx)
        @. tmp_tx -= c*tau
        nres_tx = norm(tmp_tx)*invtau

        # tmp_ty = A*tx - b*tau
        mul!(tmp_ty, A, tx)
        nres_y = norm(tmp_ty)
        @. tmp_ty -= b*tau
        nres_ty = norm(tmp_ty)*invtau

        # tmp_tz = ts + G*tx - h*tau
        mul!(tmp_tz, G, tx)
        @. tmp_tz += ts
        nres_z = norm(tmp_tz)
        @. tmp_tz -= h*tau
        nres_tz = norm(tmp_tz)*invtau

        (cx, by, hz) = (dot(c, tx), dot(b, ty), dot(h, tz))
        obj_pr = cx*invtau
        obj_du = -(by + hz)*invtau
        gap = dot(tz, ts) # TODO maybe should adapt original Alfonso condition instead of using this CVXOPT condition

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

        if opt.verbose
            # print iteration statistics
            @printf("%5d %12.4e %12.4e %9.2e %9.2e %9.2e %9.2e %9.2e %9.2e\n", iter, obj_pr, obj_du, relgap, nres_pr, nres_du, tau, kap, mu)
            flush(stdout)
        end

        # check convergence criteria
        # TODO nearly primal or dual infeasible or nearly optimal cases?
        if nres_pr <= opt.tolfeas && nres_du <= opt.tolfeas && (gap <= opt.tolabsopt || (!isnan(relgap) && relgap <= opt.tolrelopt))
            opt.verbose && println("optimal solution found; terminating")
            opt.status = :Optimal

            opt.x = tx .*= invtau
            opt.s = ts .*= invtau
            opt.y = ty .*= invtau
            opt.z = tz .*= invtau
            break
        elseif !isnan(infres_pr) && infres_pr <= opt.tolfeas
            opt.verbose && println("primal infeasibility detected; terminating")
            opt.status = :PrimalInfeasible

            invobj = inv(-by - hz)
            opt.x = tx .= NaN
            opt.s = ts .= NaN
            opt.y = ty .*= invobj
            opt.z = tz .*= invobj
            break
        elseif !isnan(infres_du) && infres_du <= opt.tolfeas
            opt.verbose && println("dual infeasibility detected; terminating")
            opt.status = :DualInfeasible

            invobj = inv(-cx)
            opt.x = tx .*= invobj
            opt.s = ts .*= invobj
            opt.y = ty .= NaN
            opt.z = tz .= NaN
            break
        elseif mu <= opt.tolfeas*1e-2 && tau <= opt.tolfeas*1e-2*min(1.0, kap)
            opt.verbose && println("ill-posedness detected; terminating")
            opt.status = :IllPosed

            opt.x = tx .= NaN
            opt.s = ts .= NaN
            opt.y = ty .= NaN
            opt.z = tz .= NaN
            break
        end

        # check iteration limit
        iter += 1
        if iter >= opt.maxiter
            opt.verbose && println("iteration limit reached; terminating")
            opt.status = :IterationLimit
            break
        end

        # prediction phase
        # calculate prediction direction
        @. tmp_ts = tmp_tz
        @. tmp_tz = -tz
        calcHarr!(H, Matrix(mu*I, q, q), cone) # TODO rewrite function to just return the H (or vector of H_k)
        (tmp_kap, tmp_tau) = solvelinsys6!(tmp_tx, tmp_ty, tmp_tz, tmp_ts, -kap, kap + cx + by + hz, mu, tau, H, opt.L)

        # determine step length alpha by line search
        alpha = alphapred
        nbhd = Inf
        alphaprevok = true
        predfail = false
        nprediters = 0
        while true
            nprediters += 1

            @. ls_ts = ts + alpha*tmp_ts

            # accept primal iterate if
            # - decreased alpha and it is the first inside the cone and beta-neighborhood or
            # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
            if incone(cone)
                # primal iterate is inside the cone

                @. ls_tz = tz + alpha*tmp_tz
                ls_tk = (tau + alpha*tmp_tau)*(kap + alpha*tmp_kap)
                ls_mu = (dot(ls_ts, ls_tz) + ls_tk)/bnu
                nbhd = calcnbhd(ls_tk, ls_mu, ls_tz, g, cone)

                if nbhd < abs2(beta*ls_mu)
                    # iterate is inside the beta-neighborhood
                    if !alphaprevok || (alpha > opt.predlsmulti)
                        # either the previous iterate was outside the beta-neighborhood or increasing alpha again will make it > 1
                        if opt.predlinesearch
                            alphapred = alpha
                        end
                        break
                    end

                    alphaprevok = true
                    alpha = alpha/opt.predlsmulti # increase alpha
                    continue
                end
            end

            # primal iterate is either
            # - outside the cone or
            # - inside the cone and outside the beta-neighborhood and previous iterate was outside the beta-neighborhood
            if alpha < alphapredthres
                # alpha is very small, so predictor has failed
                predfail = true
                opt.verbose && println("predictor could not improve the solution ($nprediters line search steps); terminating")
                opt.status = :PredictorFail
                break
            end

            alphaprevok = false
            alpha = opt.predlsmulti*alpha # decrease alpha
        end
        if predfail
            break
        end

        # step distance alpha in the direction
        @. tx += alpha*tmp_tx
        @. ty += alpha*tmp_ty
        @. tz += alpha*tmp_tz
        @. ts = ls_ts
        tau += alpha*tmp_tau
        kap += alpha*tmp_kap
        mu = (dot(ts, tz) + tau*kap)/bnu

        # skip correction phase if allowed and current iterate is in the eta-neighborhood
        if opt.corrcheck && (nbhd < abs2(eta*mu))
            continue
        end

        # correction phase
        corrfail = false
        ncorrsteps = 0
        while true
            ncorrsteps += 1

            # calculate correction direction
            @. tmp_tx = 0.0
            @. tmp_ty = 0.0
            calcg!(g, cone)
            @. tmp_tz = -tz - mu*g
            @. tmp_ts = 0.0
            calcHarr!(H, Matrix(mu*I, q, q), cone) # TODO rewrite function to just return the H (or vector of H_k)
            (tmp_kap, tmp_tau) = solvelinsys6!(tmp_tx, tmp_ty, tmp_tz, tmp_ts, -kap + mu/tau, 0.0, mu, tau, H, opt.L)

            # determine step length alpha by line search
            alpha = opt.alphacorr
            ncorrlsiters = 0
            while ncorrlsiters <= opt.maxcorrlsiters
                ncorrlsiters += 1

                @. ls_ts = ts + alpha*tmp_ts
                if incone(cone)
                    # primal iterate tx is inside the cone, so terminate line search
                    break
                end

                # primal iterate tx is outside the cone
                if ncorrlsiters == opt.maxcorrlsiters
                    # corrector failed
                    corrfail = true
                    opt.verbose && println("corrector could not improve the solution ($ncorrlsiters line search steps); terminating")
                    opt.status = :CorrectorFail
                    break
                end

                alpha = opt.corrlsmulti*alpha # decrease alpha
            end
            if corrfail
                break
            end

            # step distance alpha in the direction
            @. tx += alpha*tmp_tx
            @. ty += alpha*tmp_ty
            @. tz += alpha*tmp_tz
            @. ts = ls_ts
            tau += alpha*tmp_tau
            kap += alpha*tmp_kap
            mu = (dot(ts, tz) + tau*kap)/bnu

            # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
            if (ncorrsteps == opt.maxcorrsteps) || opt.corrcheck
                @. ls_tz = tz
                nbhd = calcnbhd(tau*kap, mu, ls_tz, g, cone)
                if nbhd <= abs2(eta*mu)
                    break
                elseif ncorrsteps == opt.maxcorrsteps
                    # outside eta neighborhood, so corrector failed
                    corrfail = true
                    opt.verbose && println("corrector phase finished outside the eta-neighborhood ($ncorrsteps correction steps); terminating")
                    opt.status = :CorrectorFail
                    break
                end
            end
        end
        if corrfail
            break
        end
    end

    opt.verbose && println("\nfinished in $iter iterations; internal status is $(opt.status)\n")

    # calculate solution and iteration statistics
    opt.niters = iter
    opt.tau = tau
    opt.kap = kap
    opt.mu = mu
    opt.solvetime = time() - starttime

    return nothing
end

# TODO put this inside the cone functions
function calcnbhd(tk, mu, ls_tz, g, cone)
    calcg!(g, cone)
    @. ls_tz += mu*g
    calcHiarr!(g, ls_tz, cone)
    return (tk - mu)^2 + dot(ls_tz, g)
end

# get neighborhood parameters depending on magnitude of barrier parameter and maximum number of correction steps
function getbetaeta(maxcorrsteps::Int, bnu::Float64)
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
