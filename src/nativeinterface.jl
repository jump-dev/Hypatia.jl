
mutable struct LinSysCache
    Q1
    Q2
    Ri
    HG
    GHG
    GHGQ2
    Q2GHGQ2
    bxGHbz
    Riby
    Q1x
    rhs
    Q2div
    Q2x
    GHGxi
    Q1yirhs
    HGxi
    x1
    y1
    z1

    function LinSysCache(Q1::Matrix{Float64}, Q2::Matrix{Float64}, Ri::AbstractMatrix{Float64}, n::Int, p::Int, q::Int)
        L = new()
        nmp = n - p
        @assert size(Q1) == (n, p)
        @assert size(Q2) == (n, nmp)
        @assert size(Ri) == (p, p)
        L.Q1 = Q1
        L.Q2 = Q2
        L.Ri = Ri
        L.HG = Matrix{Float64}(undef, q, n)
        L.GHG = Matrix{Float64}(undef, n, n)
        L.GHGQ2 = Matrix{Float64}(undef, n, nmp)
        L.Q2GHGQ2 = Matrix{Float64}(undef, nmp, nmp)
        L.bxGHbz = Vector{Float64}(undef, n)
        L.Riby = Vector{Float64}(undef, p)
        L.Q1x = Vector{Float64}(undef, n)
        L.rhs = Vector{Float64}(undef, n)
        L.Q2div = Vector{Float64}(undef, nmp)
        L.Q2x = Vector{Float64}(undef, n)
        L.GHGxi = Vector{Float64}(undef, n)
        L.Q1yirhs = Vector{Float64}(undef, p)
        L.HGxi = Vector{Float64}(undef, q)
        L.x1 = Vector{Float64}(undef, n)
        L.y1 = Vector{Float64}(undef, p)
        L.z1 = Vector{Float64}(undef, q)
        return L
    end
end

"""
solves a pair of primal and dual cone programs
 primal (over x,s):
  min  c'x :          duals
    b - Ax == 0       (y)
    h - Gx == s in K  (z)
 dual (over z,y):
  max  -b'y - h'z :      duals
    c + A'y + G'z == 0   (x)
                z in K*  (s)
where K is a convex cone defined as a Cartesian product of recognized primitive cones, and K* is its dual cone

the primal-dual optimality conditions are
           b - Ax == 0
           h - Gx == s
    c + A'y + G'z == 0
              s'z == 0
                s in K
                z in K*
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
    cone::Cone                  # primal constraint cone object

    L::LinSysCache  # cache for direction finding functions


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
    maxpredsmallsteps = 15,
    maxcorrsteps = 15,
    corrcheck = true,
    maxcorrlsiters = 15,
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
        error("maxpredsmallsteps must be at least 1")
    end
    if maxcorrsteps < 1
        error("maxcorrsteps must be at least 1")
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
    cone::Cone;
    check::Bool=false, # check rank conditions
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
    # TODO reduce allocs, improve efficiency
    # A' = [Q1 Q2] * [R1; 0]
    if issparse(A)
        alf.verbose && println("\nJulia is currently missing some sparse matrix methods that could improve performance; Alfonso may perform better if A is loaded as a dense matrix")
        # TODO currently using dense Q1, Q2, R - probably some should be sparse
        F = qr(sparse(A'))
        @assert length(F.prow) == n
        @assert length(F.pcol) == p
        @assert istriu(F.R)

        Q = F.Q*Matrix(1.0I, n, n)
        Q1 = zeros(n, p)
        Q1[F.prow, F.pcol] = Q[:, 1:p]
        Q2 = zeros(n, n-p)
        Q2[F.prow, :] = Q[:, p+1:n]
        Ri = zeros(p, p)
        Ri[F.pcol, F.pcol] = inv(UpperTriangular(F.R))
    else
        F = qr(A')
        @assert istriu(F.R)

        Q = F.Q*Matrix(1.0I, n, n)
        Q1 = Q[:, 1:p]
        Q2 = Q[:, p+1:n]
        Ri = inv(UpperTriangular(F.R))
        @assert norm(A'*Ri - Q1) < 1e-8 # TODO delete later
    end

    if check
        # check rank conditions
        # TODO rank for qr decomp should be implemented in Julia - see https://github.com/JuliaLang/julia/blob/f8b52dab77415a22d28497f48407aca92fbbd4c3/stdlib/LinearAlgebra/src/qr.jl#L895
        if rank(A) < p # TODO change to rank(F)
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
    alf.cone = cone
    alf.L = LinSysCache(Q1, Q2, Ri, n, p, q)
    alf.status = :Loaded

    return alf
end

# solve using homogeneous self-dual embedding
function solve!(alf::AlfonsoOpt)
    starttime = time()

    (c, A, b, G, h, cone) = (alf.c, alf.A, alf.b, alf.G, alf.h, alf.cone)
    (n, p, q) = (length(c), length(b), length(h))
    bnu = 1.0 + barrierpar(cone) # complexity parameter nu-bar of the augmented barrier (sum of the primitive cone barrier parameters plus 1)

    # preallocate arrays
    # primal and dual variables multiplied by tau
    tx = similar(c)
    ty = similar(b)
    tz = similar(h)
    ts = similar(h)
    # gradient evaluations at ls_ts of the barrier function for K
    g = similar(ts)
    # search directions
    dir_tx = similar(tx)
    dir_ty = similar(ty)
    dir_tz = similar(tz)
    dir_ts = similar(ts)
    # values during line searches
    ls_tz = similar(tz)
    ls_ts = similar(ts)

    # cone functions evaluate barrier derivatives at ls_ts
    loadpnt!(cone, ls_ts)

    # find initial primal-dual iterate
    (tau, kap, mu) = findinitialiterate!(tx, ty, tz, ts, ls_ts, bnu, alf)

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

            invtau = inv(tau)
            alf.x = tx .*= invtau
            alf.s = ts .*= invtau
            alf.y = ty .*= invtau
            alf.z = tz .*= invtau
            break
        elseif !isnan(infres_pr) && infres_pr <= alf.tolfeas
            alf.verbose && println("primal infeasibility detected; terminating")
            alf.status = :PrimalInfeasible

            invobj = inv(-by - hz)
            alf.x = tx .= NaN
            alf.s = ts .= NaN
            alf.y = ty .*= invobj
            alf.z = tz .*= invobj
            break
        elseif !isnan(infres_du) && infres_du <= alf.tolfeas
            alf.verbose && println("dual infeasibility detected; terminating")
            alf.status = :DualInfeasible

            invobj = inv(-cx)
            alf.x = tx .*= invobj
            alf.s = ts .*= invobj
            alf.y = ty .= NaN
            alf.z = tz .= NaN
            break
        elseif mu <= alf.tolfeas*1e-2 && tau <= alf.tolfeas*1e-2*min(1.0, kap)
            alf.verbose && println("ill-posedness detected; terminating")
            alf.status = :IllPosed

            alf.x = tx .= NaN
            alf.s = ts .= NaN
            alf.y = ty .= NaN
            alf.z = tz .= NaN
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
        @. dir_tx = res_tx
        @. dir_ty = res_ty
        @. dir_tz = -tz
        @. dir_ts = res_tz
        (dir_kap, dir_tau) = finddirection!(dir_tx, dir_ty, dir_tz, dir_ts, -kap, res_tau, mu, tau, alf)

        # determine step length alpha by line search
        alpha = alphapred
        nbhd = Inf
        alphaprevok = true
        predfail = false
        nprediters = 0
        while true
            nprediters += 1

            @. ls_ts = ts + alpha*dir_ts

            # accept primal iterate if
            # - decreased alpha and it is the first inside the cone and beta-neighborhood or
            # - increased alpha and it is inside the cone and the first to leave beta-neighborhood
            if incone(cone)
                # primal iterate is inside the cone
                @. ls_tz = tz + alpha*dir_tz
                ls_tk = (tau + alpha*dir_tau)*(kap + alpha*dir_kap)
                ls_mu = (dot(ls_ts, ls_tz) + ls_tk)/bnu
                nbhd = calcnbhd(ls_tk, ls_mu, ls_tz, g, cone)

                if nbhd < abs2(beta*ls_mu)
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
        @. tx += alpha*dir_tx
        @. ty += alpha*dir_ty
        @. tz += alpha*dir_tz
        @. ts = ls_ts
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
            @. dir_tx = 0.0
            @. dir_ty = 0.0
            calcg!(g, cone)
            @. dir_tz = -tz - mu*g
            @. dir_ts = 0.0
            (dir_kap, dir_tau) = finddirection!(dir_tx, dir_ty, dir_tz, dir_ts, -kap + mu/tau, 0.0, mu, tau, alf)

            # determine step length alpha by line search
            alpha = alf.alphacorr
            ncorrlsiters = 0
            while ncorrlsiters <= alf.maxcorrlsiters
                ncorrlsiters += 1

                @. ls_ts = ts + alpha*dir_ts
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
            @. tx += alpha*dir_tx
            @. ty += alpha*dir_ty
            @. tz += alpha*dir_tz
            @. ts = ls_ts
            tau += alpha*dir_tau
            kap += alpha*dir_kap
            mu = (dot(ts, tz) + tau*kap)/bnu

            # finish if allowed and current iterate is in the eta-neighborhood, or if taken max steps
            if (ncorrsteps == alf.maxcorrsteps) || alf.corrcheck
                @. ls_tz = tz
                nbhd = calcnbhd(tau*kap, mu, ls_tz, g, cone)
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
    alf.tau = tau
    alf.kap = kap
    alf.mu = mu
    alf.solvetime = time() - starttime

    return nothing
end

# TODO put this inside the cone functions
function calcnbhd(tk, mu, ls_tz, g, cone)
    calcg!(g, cone)
    @. ls_tz += mu*g
    calcHiarr!(g, ls_tz, cone)
    return (tk - mu)^2 + dot(ls_tz, g)
end

# calculate initial central primal-dual iterate
function findinitialiterate!(tx, ty, tz, ts, ls_ts, bnu, alf)
    (b, G, h, cone) = (alf.b, alf.G, alf.h, alf.cone)
    L = alf.L
    (Q1, Q2, Ri, Q2GHGQ2, bxGHbz, Riby, Q1x, rhs, Q2div, Q2x, Q1yirhs) = (L.Q1, L.Q2, L.Ri, L.Q2GHGQ2, L.bxGHbz, L.Riby, L.Q1x, L.rhs, L.Q2div, L.Q2x, L.Q1yirhs)

    alf.verbose && println("\nfinding initial iterate")

    # solve linear equation
    # |0  A' G'| * |tx| = |0|
    # |A  0  0 |   |ty|   |b|
    # |G  0  -I|   |ts|   |h|

    # GQ2 = G*Q2
    # Q1x = Q1*Ri'*b
    # Q2x = Q2*(Symmetric(GQ2'*GQ2)\(GQ2'*(h - G*Q1x)))
    # tx = Q1x .+ Q2x
    # ts = h - G*tx
    # ty = Ri*Q1'*G'*ts

    GQ2 = G*Q2 # TODO not prealloced
    mul!(Q2GHGQ2, GQ2', GQ2)
    F = bunchkaufman!(Symmetric(Q2GHGQ2))

    # Q1x = Q1*Ri'*b
    mul!(Riby, Ri', b)
    mul!(Q1x, Q1, Riby)
    # Q2x = Q2*(F\(GQ2'*(h - G*Q1x)))
    mul!(rhs, G, Q1x)
    @. rhs = h - rhs
    mul!(Q2div, GQ2', rhs)
    ldiv!(F, Q2div)
    mul!(Q2x, Q2, Q2div)
    # tx = Q1x + Q2x
    @. tx = Q1x + Q2x
    # ts = h - G*tx
    mul!(ts, G, tx)
    @. ts = h - ts
    # ty = Ri*Q1'*G'*ts
    mul!(bxGHbz, G', ts)
    mul!(Q1yirhs, Q1', bxGHbz)
    mul!(ty, Ri, Q1yirhs)

    # from ts, step along interior direction of cone until ts is inside cone
    @. ls_ts = ts
    if !incone(cone)
        dir_ts = getintdir!(rhs, cone)
        alpha = 1.0 # TODO starting alpha maybe should depend on ls_ts (eg norm like in Alfonso) in case 1.0 is too large/small
        steps = 0
        while !incone(cone)
            @. ls_ts = ts + alpha*dir_ts
            alpha *= 1.5
            steps += 1
            if steps > 25
                error("cannot find initial iterate")
            end
        end
        alf.verbose && println("$steps steps taken for initial iterate")
        @. ts = ls_ts
    end

    @assert incone(cone) # TODO delete
    calcg!(tz, cone)
    @. tz *= -1.0

    tau = 1.0
    kap = 1.0
    mu = (dot(tz, ts) + tau*kap)/bnu

    # TODO delete later
    @assert abs(1.0 - mu) < 1e-8
    @assert calcnbhd(tau*kap, mu, copy(tz), copy(tz), cone) < 1e-6

    alf.verbose && println("initial iterate found")

    return (tau, kap, mu)
end



function finddirection!(rhs_tx, rhs_ty, rhs_tz, rhs_ts, rhs_kap, rhs_tau, mu, tau, alf)
    (c, A, b, G, h, cone) = (alf.c, alf.A, alf.b, alf.G, alf.h, alf.cone)
    (n, p, q) = (length(c), length(b), length(h))
    L = alf.L
    (Q1, Q2, Ri, HG, GHG, GHGQ2, Q2GHGQ2, x1, y1, z1) = (L.Q1, L.Q2, L.Ri, L.HG, L.GHG, L.GHGQ2, L.Q2GHGQ2, L.x1, L.y1, L.z1)

    # solve two symmetric systems and combine the solutions
    # use QR + cholesky method from CVXOPT
    # (1) eliminate equality constraints via QR of A'
    # (2) solve reduced system by cholesky
    # |0  A'  G'   | * |ux| = |bx|
    # |A  0   0    |   |uy|   |by|
    # |G  0  -Hi/mu|   |uz|   |bz|

    # A' = [Q1 Q2] * [R1; 0]
    # factorize Q2' * G' * mu*H * G * Q2
    calcHarr!(HG, G, cone)
    @. HG *= mu
    mul!(GHG, G', HG)
    mul!(GHGQ2, GHG, Q2)
    mul!(Q2GHGQ2, Q2', GHGQ2)

    # TODO cholesky vs bunch-kaufman?
    F = bunchkaufman!(Symmetric(Q2GHGQ2))
    # F = cholesky!(Symmetric(Q2GHGQ2))

    # (x2, y2, z2) = (rhs_tx, -rhs_ty, -mu*H*rhs_ts - rhs_tz)
    @. rhs_ty *= -1.0
    @. rhs_tz *= -1.0
    if !iszero(rhs_ts)
        calcHarr!(z1, rhs_ts, cone)
        @. rhs_tz -= mu*z1
    end
    calcxyz!(rhs_tx, rhs_ty, rhs_tz, F, alf)

    # (x1, y1, z1) = (-c, b, mu*H*h)
    @. x1 = -c
    @. y1 = b
    calcHarr!(z1, h, cone)
    @. z1 *= mu
    calcxyz!(x1, y1, z1, F, alf)

    # combine
    dir_tau = (rhs_tau + rhs_kap + dot(c, rhs_tx) + dot(b, rhs_ty) + dot(h, rhs_tz))/(mu/tau/tau - dot(c, x1) - dot(b, y1) - dot(h, z1))
    @. rhs_tx += dir_tau*x1
    @. rhs_ty += dir_tau*y1
    @. rhs_tz += dir_tau*z1
    mul!(z1, G, rhs_tx)
    @. rhs_ts = -z1 + h*dir_tau - rhs_ts
    dir_kap = -dot(c, rhs_tx) - dot(b, rhs_ty) - dot(h, rhs_tz) - rhs_tau

    return (dir_kap, dir_tau)
end

function calcxyz!(xi, yi, zi, F, alf)
    L = alf.L
    (Q1, Q2, Ri, HG, GHG, bxGHbz, Riby, Q1x, rhs, Q2div, Q2x, GHGxi, Q1yirhs, HGxi) = (L.Q1, L.Q2, L.Ri, L.HG, L.GHG, L.bxGHbz, L.Riby, L.Q1x, L.rhs, L.Q2div, L.Q2x, L.GHGxi, L.Q1yirhs, L.HGxi)

    # bxGHbz = bx + G'*Hbz
    mul!(bxGHbz, alf.G', zi)
    @. bxGHbz += xi
    # Q1x = Q1*Ri'*by
    mul!(Riby, Ri', yi)
    mul!(Q1x, Q1, Riby)
    # Q2x = Q2*(K22_F\(Q2'*(bxGHbz - GHG*Q1x)))
    mul!(rhs, GHG, Q1x)
    @. rhs = bxGHbz - rhs
    mul!(Q2div, Q2', rhs)
    ldiv!(F, Q2div)
    mul!(Q2x, Q2, Q2div)
    # xi = Q1x + Q2x
    @. xi = Q1x + Q2x
    # yi = Ri*Q1'*(bxGHbz - GHG*xi)
    mul!(GHGxi, GHG, xi)
    @. bxGHbz -= GHGxi
    mul!(Q1yirhs, Q1', bxGHbz)
    mul!(yi, Ri, Q1yirhs)
    # zi = HG*xi - Hbz
    mul!(HGxi, HG, xi)
    @. zi = HGxi - zi

    return (xi, yi, zi)
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
