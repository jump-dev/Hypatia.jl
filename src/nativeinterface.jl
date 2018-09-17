#=
reference V. is Vandenberghe at http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf

quadratic objective (P is PSD) with scalar offset o (S1 of V.)
 primal (over x):
  min  1/2 x'Px + c'x + o :
               b - Ax == 0        (y)
               h - Gx == s in K   (z)
 dual (over z,y,w):
  max  -1/2 w'Pw - b'y - h'z + o :
               c + A'y + G'z == Pw   (x)
                           z in K*   (s)

optimality conditions are:
  c + Px + A'y + G'z == 0
  b - Ax             == 0
  h - Gx             == s
 and:
  z's == 0
    s in K
    z in K*
=#

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
    P::AbstractMatrix{Float64}  # quadratic cost matrix, size n*n
    c::Vector{Float64}          # linear cost vector, size n
    o::Float64                  # objective offset scalar
    A::AbstractMatrix{Float64}  # equality constraint matrix, size p*n
    b::Vector{Float64}          # equality constraint vector, size p
    G::AbstractMatrix{Float64}  # cone constraint matrix, size q*n
    h::Vector{Float64}          # cone constraint vector, size q
    cone::Cone                  # primal cone object, size q
    bnu::Float64                # complexity parameter nu-bar of the augmented barrier (sum of the primitive cone barrier parameters plus 1)

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

# load and verify problem data, calculate algorithmic parameters
function load_data!(
    alf::AlfonsoOpt,
    P::AbstractMatrix{Float64},
    c::Vector{Float64},
    o::Float64,
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cone;
    check=true, # TODO later make false
    )

    # check data consistency
    if check
        n = length(c)
        p = length(b)
        q = length(h)
        @assert n > 0
        @assert p + q > 0
        if n != size(A, 2) || n != size(G, 2) || n != size(P, 2) || n != size(P, 1)
            error("number of variables is not consistent in Q, A, G, and c")
        end
        if p != size(A, 1)
            error("number of constraint rows is not consistent in A and b")
        end
        if q != size(G, 1)
            error("number of constraint rows is not consistent in G and h")
        end
        if eigmin(P) < 0.0 # TODO slow
            error("P matrix is not positive semidefinite")
        end

        # TODO do appropriate decomps at the same time, do preprocessing
        if rank(A) < p
            error("A matrix is not full-row-rank; some primal equalities may be redundant or inconsistent")
        end
        if rank([P A' G']) < n
            error("[P A' G'] is not full-row-rank; some dual equalities may be redundant (i.e. primal variables can be removed) or inconsistent")
        end
    end

    if issparse(P)
        dropzeros!(P)
    end
    if issparse(A)
        dropzeros!(A)
    end
    if issparse(G)
        dropzeros!(G)
    end

    # TODO check cone consistency in cone functions file

    # save data in solver object
    alf.P = P
    alf.c = c
    alf.o = o
    alf.A = A
    alf.b = b
    alf.G = G
    alf.h = h
    alf.cone = cone
    alf.bnu = 1.0 + barrierpar(cone)
    alf.status = :Loaded

    return alf
end

# TODO put in new file for linsys solvers, use a cache for each
# function solvelinsys(y1, x1, y2, x2, mu, rhs_tx, rhs_ty, A, b, c, cone, HiAt, AHiAt)
#     invmu = inv(mu)
#
#     # TODO could ultimately be faster or more stable to do cholesky.L ldiv everywhere than to do full hessian ldiv
#     calcHiarr!(HiAt, A', cone)
#     HiAt .*= invmu
#     mul!(AHiAt, A, HiAt)
#     F = cholesky!(Symmetric(AHiAt))
#
#     # TODO can parallelize 1 and 2
#     # y2 = F\(rhs_ty + HiAt'*rhs_tx)
#     mul!(y2, HiAt', rhs_tx)
#     y2 .+= rhs_ty
#     ldiv!(F, y2) # y2 done
#
#     # x2 = Hi*invmu*(A'*y2 - rhs_tx)
#     mul!(x2, A', y2)
#     rhs_tx .= x2 .- rhs_tx # destroys rhs_tx
#     rhs_tx .*= invmu
#     calcHiarr!(x2, rhs_tx, cone) # x2 done
#
#     # y1 = F\(b + HiAt'*c)
#     mul!(y1, HiAt', c)
#     y1 .+= b
#     ldiv!(F, y1) # y1 done
#
#     # x1 = Hi*invmu*(A'*y1 - c)
#     mul!(rhs_tx, A', y1)
#     rhs_tx .-= c
#     rhs_tx .*= invmu
#     calcHiarr!(x1, rhs_tx, cone) # x1 done
#
#     return (y1, x1, y2, x2)
# end

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
