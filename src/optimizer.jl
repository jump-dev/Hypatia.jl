

mutable struct AlfonsoOptimizer <: MOI.AbstractOptimizer
    # options
    verbose::Bool           # if true, prints progress at each iteration
    optimtol::Float64       # optimization tolerance parameter
    maxiter::Int            # maximum number of iterations
    predlinesearch::Bool    # if false, predictor step uses a fixed step size, else step size is determined via line search
    maxpredsmallsteps::Int  # maximum number of predictor step size reductions allowed with respect to the safe fixed step size
    maxcorrsteps::Int       # maximum number of corrector steps (possible values: 1, 2, or 4)
    corrcheck::Bool         # if false, maxcorrsteps corrector steps are performed at each corrector phase, else the corrector phase can be terminated before maxcorrsteps corrector steps if the iterate is in the eta-neighborhood
    maxcorrlsiters::Int     # maximum number of line search iterations in each corrector step
    maxitrefinesteps::Int   # maximum number of iterative refinement steps in linear system solves
    alphacorr::Float64      # corrector step size
    predlsmulti::Float64    # predictor line search step size multiplier
    corrlsmulti::Float64    # corrector line search step size multiplier
    itrefinethres::Float64  # iterative refinement success threshold

    # problem data
    A::AbstractMatrix{Float64}          # constraint matrix
    b::Vector{Float64}                  # right-hand side vector
    c::Vector{Float64}                  # cost vector
    cones::Vector{ConeData}             # primal cones list
    coneidxs::Vector{AbstractUnitRange} # primal cones variable indices list

    # results
    status::Symbol          # solver status
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

    function AlfonsoOptimizer(verbose, optimtol, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, maxitrefinesteps, alphacorr, predlsmulti, corrlsmulti, itrefinethres)
        opt = new()

        opt.verbose = verbose
        opt.optimtol = optimtol
        opt.maxiter = maxiter
        opt.predlinesearch = predlinesearch
        opt.maxpredsmallsteps = maxpredsmallsteps
        opt.maxcorrsteps = maxcorrsteps
        opt.corrcheck = corrcheck
        opt.maxcorrlsiters = maxcorrlsiters
        opt.maxitrefinesteps = maxitrefinesteps
        opt.alphacorr = alphacorr
        opt.predlsmulti = predlsmulti
        opt.corrlsmulti = corrlsmulti
        opt.itrefinethres = itrefinethres

        opt.status = :NotLoaded

        return opt
    end
end


function AlfonsoOptimizer(;
    verbose = true,
    optimtol = 1e-06,
    maxiter = 1e3,
    predlinesearch = true,
    maxpredsmallsteps = 8,
    maxcorrsteps = 8, # NOTE doubled in .m code
    corrcheck = true,
    maxcorrlsiters = 8,
    maxitrefinesteps = 0,
    alphacorr = 1.0,
    predlsmulti = 0.7,
    corrlsmulti = 0.5,
    itrefinethres = 0.1,
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

    return AlfonsoOptimizer(verbose, optimtol, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, maxitrefinesteps, alphacorr, predlsmulti, corrlsmulti, itrefinethres)
end


function loaddata!(opt::AlfonsoOptimizer, A::AbstractMatrix{Float64}, b::Vector{Float64}, c::Vector{Float64}, cones::Vector{ConeData}, coneidxs::Vector{AbstractUnitRange})
    #=
    verify problem data, setup other algorithmic parameters and utilities
    TODO simple presolve (see ConicIP solver)
    =#
    (m, n) = size(A)
    if (m == 0) || (n == 0)
        error("input matrix A has trivial dimension $m x $n")
    end
    if m != length(b)
        error("dimension of vector b is $(length(b)), but number of rows in matrix A is $m")
    end
    if n != length(c)
        error("dimension of vector c is $(length(c)), but number of columns in matrix A is $n")
    end
    if issparse(A)
        dropzeros!(A)
    end

    idxend = 0
    for k in eachindex(cones)
        if dimension(cones[k]) != length(coneidxs[k])
            error("dimension of cone type $(cones[k]) does not match length of variable indices")
        end
        @assert coneidxs[k][1] == idxend + 1
        idxend += length(coneidxs[k])
    end
    @assert idxend == n

    # coneobjs = [coneobj(ck) for ck in cones] # TODO convert from MOI cones

    # save data in optimizer object
    opt.A = A
    opt.b = b
    opt.c = c
    opt.cones = cones
    opt.coneidxs = coneidxs

    opt.status = :Loaded

    return opt
end
