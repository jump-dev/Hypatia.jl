

export AlfonsoOptimizer

mutable struct AlfonsoOptimizer <: MOI.AbstractOptimizer
    # options
    verbose::Bool               # if true, prints progress at each iteration
    optimtol::Float64           # optimization tolerance parameter
    maxiter::Int                # maximum number of iterations
    predlinesearch::Bool        # if false, predictor step uses a fixed step size, else step size is determined via line search
    maxpredsmallsteps::Int      # maximum number of predictor step size reductions allowed with respect to the safe fixed step size
    maxcorrsteps::Int           # maximum number of corrector steps (possible values: 1, 2, or 4)
    corrcheck::Bool             # if false, maxcorrsteps corrector steps are performed at each corrector phase, else the corrector phase can be terminated before maxcorrsteps corrector steps if the iterate is in the eta-neighborhood
    maxcorrlsiters::Int         # maximum number of line search iterations in each corrector step
    maxitrefinesteps::Int       # maximum number of iterative refinement steps in linear system solves
    alphacorr::Float64          # corrector step size
    predlsmulti::Float64        # predictor line search step size multiplier
    corrlsmulti::Float64        # corrector line search step size multiplier
    itrefinethreshold::Float64  # iterative refinement success threshold

    # problem data
    A               # constraint matrix
    b               # right-hand side vector
    c               # cost vector
    cones           # TODO

    # other algorithmic parameters and utilities
    eval_gh::Function           # function for computing the gradient and Hessian of the barrier function
    gh_bnu::Float64             # complexity parameter of the augmented barrier (nu-bar)
    beta::Float64               # large neighborhood parameter
    eta::Float64                # small neighborhood parameter
    alphapredls::Float64        # initial predictor step size with line search
    alphapredfix::Float64       # fixed predictor step size
    alphapred::Float64          # initial predictor step size
    alphapredthreshold::Float64 # minimum predictor step size

    # results
    status          # solver status
    niterations     # total number of iterations
    all_alphapred   # predictor step size at each iteration
    all_betapred    # neighborhood parameter at the end of the predictor phase at each iteration
    all_etacorr     # neighborhood parameter at the end of the corrector phase at each iteration
    all_mu          # complementarity gap at each iteration
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

    function AlfonsoOptimizer(verbose, optimtol, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, maxitrefinesteps, alphacorr, predlsmulti, corrlsmulti, itrefinethreshold)
        mod = new()

        mod.verbose = verbose
        mod.optimtol = optimtol
        mod.maxiter = maxiter
        mod.predlinesearch = predlinesearch
        mod.maxpredsmallsteps = maxpredsmallsteps
        mod.maxcorrsteps = maxcorrsteps
        mod.corrcheck = corrcheck
        mod.maxcorrlsiters = maxcorrlsiters
        mod.maxitrefinesteps = maxitrefinesteps
        mod.alphacorr = alphacorr
        mod.predlsmulti = predlsmulti
        mod.corrlsmulti = corrlsmulti
        mod.itrefinethreshold = itrefinethreshold

        mod.status = :NotLoaded

        return mod
    end
end


function AlfonsoOptimizer(;
    verbose = false,
    optimtol = 1e-06,
    maxiter = 1e4,
    predlinesearch = true,
    maxpredsmallsteps = 8,
    maxcorrsteps = 8, # NOTE doubled in .m code
    corrcheck = true,
    maxcorrlsiters = 8,
    maxitrefinesteps = 0,
    alphacorr = 1.0,
    predlsmulti = 0.7,
    corrlsmulti = 0.5,
    itrefinethreshold = 0.1,
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

    return AlfonsoOptimizer(verbose, optimtol, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, maxitrefinesteps, alphacorr, predlsmulti, corrlsmulti, itrefinethreshold)
end
