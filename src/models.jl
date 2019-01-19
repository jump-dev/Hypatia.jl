#=
Copyright 2018, Chris Coey and contributors
Copyright 2018, David Papp, Sercan Yildiz
=#

# model object containing options, problem data, linear system cache, and solution
mutable struct Model
    # options
    verbose::Bool           # if true, prints progress at each iteration
    timelimit::Float64      # (approximate) time limit (in seconds) for algorithm in solve function
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
    cone::Cones.Cone                  # primal constraint cone object

    L::LinearSystems.LinearSystemSolver  # cache for linear system solves

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

    function Model(verbose, timelimit, tolrelopt, tolabsopt, tolfeas, maxiter, predlinesearch, maxpredsmallsteps, predlsmulti, corrcheck, maxcorrsteps, alphacorr, maxcorrlsiters, corrlsmulti)
        mdl = new()
        mdl.verbose = verbose
        mdl.timelimit = timelimit
        mdl.tolrelopt = tolrelopt
        mdl.tolabsopt = tolabsopt
        mdl.tolfeas = tolfeas
        mdl.maxiter = maxiter
        mdl.predlinesearch = predlinesearch
        mdl.maxpredsmallsteps = maxpredsmallsteps
        mdl.predlsmulti = predlsmulti
        mdl.corrcheck = corrcheck
        mdl.maxcorrsteps = maxcorrsteps
        mdl.alphacorr = alphacorr
        mdl.maxcorrlsiters = maxcorrlsiters
        mdl.corrlsmulti = corrlsmulti
        mdl.status = :NotLoaded
        return mdl
    end
end

# initialize a model object
function Model(;
    verbose = false,
    timelimit = 3.6e3, # TODO should be Inf
    tolrelopt = 1e-6,
    tolabsopt = 1e-7,
    tolfeas = 1e-7,
    maxiter = 1e4,
    predlinesearch = true,
    maxpredsmallsteps = 15,
    predlsmulti = 0.7,
    corrcheck = true,
    maxcorrsteps = 15,
    alphacorr = 1.0,
    maxcorrlsiters = 15,
    corrlsmulti = 0.5,
    )
    if min(tolrelopt, tolabsopt, tolfeas) < 1e-12 || max(tolrelopt, tolabsopt, tolfeas) > 1e-2
        error("tolrelopt, tolabsopt, tolfeas must be between 1e-12 and 1e-2")
    end
    if timelimit < 1e-2
        error("timelimit must be at least 1e-2")
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

    return Model(verbose, timelimit, tolrelopt, tolabsopt, tolfeas, maxiter, predlinesearch, maxpredsmallsteps, predlsmulti, corrcheck, maxcorrsteps, alphacorr, maxcorrlsiters, corrlsmulti)
end

get_status(mdl::Model) = mdl.status
get_solvetime(mdl::Model) = mdl.solvetime
get_niters(mdl::Model) = mdl.niters

get_x(mdl::Model) = copy(mdl.x)
get_s(mdl::Model) = copy(mdl.s)
get_y(mdl::Model) = copy(mdl.y)
get_z(mdl::Model) = copy(mdl.z)

get_tau(mdl::Model) = mdl.tau
get_kappa(mdl::Model) = mdl.kappa
get_mu(mdl::Model) = mdl.mu

get_pobj(mdl::Model) = dot(mdl.c, mdl.x)
get_dobj(mdl::Model) = -dot(mdl.b, mdl.y) - dot(mdl.h, mdl.z)

# check data for consistency
function check_data(
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cones.Cone,
    )
    (n, p, q) = (length(c), length(b), length(h))
    if n == 0
        println("no variables were specified; proceeding anyway")
    end
    if q == 0
        println("no conic constraints were specified; proceeding anyway")
    end
    if n < p
        println("number of equality constraints ($p) exceeds number of variables ($n)")
    end
    if n != size(A, 2) || n != size(G, 2)
        error("number of variables is not consistent in A, G, and c")
    end
    if p != size(A, 1)
        error("number of constraint rows is not consistent in A and b")
    end
    if q != size(G, 1)
        error("number of constraint rows is not consistent in G and h")
    end

    if length(cone.prmtvs) != length(cone.idxs)
        error("number of primitive cones does not match number of index ranges")
    end
    qcone = 0
    for k in eachindex(cone.prmtvs)
        if Cones.dimension(cone.prmtvs[k]) != length(cone.idxs[k])
            error("dimension of cone $k does not match number of indices in the corresponding range")
        end
        qcone += Cones.dimension(cone.prmtvs[k])
    end
    if qcone != q
        error("dimension of cone is not consistent with number of rows in G and h")
    end

    return
end

# verify problem data and load into model object
function load_data!(
    mdl::Model,
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cones.Cone,
    L::LinearSystems.LinearSystemSolver, # linear system solver cache (see linsyssolvers folder)
    )
    (mdl.c, mdl.A, mdl.b, mdl.G, mdl.h, mdl.cone, mdl.L) = (c, A, b, G, h, cone, L)
    mdl.status = :Loaded
    return mdl
end
