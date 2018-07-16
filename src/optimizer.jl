

export AlfonsoOptimizer

mutable struct AlfonsoOptimizer <: MOI.AbstractOptimizer
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

    if !(1 <= maxcorrsteps <= 8)
        error("maxcorrsteps must be from 1 to 8")
    end

    return AlfonsoOptimizer(verbose, optimtol, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, maxitrefinesteps, alphacorr, predlsmulti, corrlsmulti, itrefinethreshold)
end
