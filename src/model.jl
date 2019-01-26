#=
Copyright 2018, Chris Coey and contributors
=#

# model object containing options, problem data, linear system cache, and solution
mutable struct Model
    # options
    verbose::Bool           # if true, prints progress at each iteration
    time_limit::Float64      # (approximate) time limit (in seconds) for algorithm in solve function
    tol_rel_opt::Float64      # relative optimality gap tolerance
    tol_abs_opt::Float64      # absolute optimality gap tolerance
    tol_feas::Float64        # feasibility tolerance
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
    cone::Cones.Cone            # primal constraint cone object

    L::LinearSystems.LinearSystemSolver  # cache for linear system solves

    # results
    status::Symbol          # solver status
    solve_time::Float64      # total solve time
    niters::Int             # total number of iterations

    x::Vector{Float64}      # final value of the primal free variables
    s::Vector{Float64}      # final value of the primal cone variables
    y::Vector{Float64}      # final value of the dual free variables
    z::Vector{Float64}      # final value of the dual cone variables
    tau::Float64            # final value of the tau variable
    kap::Float64            # final value of the kappa variable
    mu::Float64             # final value of mu
    primal_obj::Float64           # final primal objective value
    dual_obj::Float64           # final dual objective value

    function Model(verbose, time_limit, tol_rel_opt, tol_abs_opt, tol_feas, maxiter, predlinesearch, maxpredsmallsteps, predlsmulti, corrcheck, maxcorrsteps, alphacorr, maxcorrlsiters, corrlsmulti)
        model = new()
        model.verbose = verbose
        model.time_limit = time_limit
        model.tol_rel_opt = tol_rel_opt
        model.tol_abs_opt = tol_abs_opt
        model.tol_feas = tol_feas
        model.maxiter = maxiter
        model.predlinesearch = predlinesearch
        model.maxpredsmallsteps = maxpredsmallsteps
        model.predlsmulti = predlsmulti
        model.corrcheck = corrcheck
        model.maxcorrsteps = maxcorrsteps
        model.alphacorr = alphacorr
        model.maxcorrlsiters = maxcorrlsiters
        model.corrlsmulti = corrlsmulti
        model.status = :NotLoaded
        return model
    end
end

# initialize a model object
function Model(;
    verbose = false,
    time_limit = 3.6e3, # TODO should be Inf
    tol_rel_opt = 1e-6,
    tol_abs_opt = 1e-7,
    tol_feas = 1e-7,
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
    if min(tol_rel_opt, tol_abs_opt, tol_feas) < 1e-12 || max(tol_rel_opt, tol_abs_opt, tol_feas) > 1e-2
        error("tol_rel_opt, tol_abs_opt, tol_feas must be between 1e-12 and 1e-2")
    end
    if time_limit < 1e-2
        error("time_limit must be at least 1e-2")
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
    return Model(verbose, time_limit, tol_rel_opt, tol_abs_opt, tol_feas, maxiter, predlinesearch, maxpredsmallsteps, predlsmulti, corrcheck, maxcorrsteps, alphacorr, maxcorrlsiters, corrlsmulti)
end

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

    if length(cone.cones) != length(cone.idxs)
        error("number of primitive cones does not match number of index ranges")
    end
    qcone = 0
    for k in eachindex(cone.cones)
        if Cones.dimension(cone.cones[k]) != length(cone.idxs[k])
            error("dimension of cone $k does not match number of indices in the corresponding range")
        end
        qcone += Cones.dimension(cone.cones[k])
    end
    if qcone != q
        error("dimension of cone is not consistent with number of rows in G and h")
    end
    return
end

# verify problem data and load into model object
function load_data!(
    model::Model,
    c::Vector{Float64},
    A::AbstractMatrix{Float64},
    b::Vector{Float64},
    G::AbstractMatrix{Float64},
    h::Vector{Float64},
    cone::Cones.Cone,
    L::LinearSystems.LinearSystemSolver,
    )
    (model.c, model.A, model.b, model.G, model.h, model.cone, model.L) = (c, A, b, G, h, cone, L)
    model.status = :Loaded
    return model
end

get_status(model::Model) = model.status
get_solve_time(model::Model) = model.solve_time
get_num_iters(model::Model) = model.niters

get_x(model::Model) = copy(model.x)
get_s(model::Model) = copy(model.s)
get_y(model::Model) = copy(model.y)
get_z(model::Model) = copy(model.z)

get_tau(model::Model) = model.tau
get_kappa(model::Model) = model.kappa
get_mu(model::Model) = model.mu

get_primal_obj(model::Model) = dot(model.c, model.x)
get_dual_obj(model::Model) = -dot(model.b, model.y) - dot(model.h, model.z)
