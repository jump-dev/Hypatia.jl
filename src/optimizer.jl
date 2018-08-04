

mutable struct Optimizer <: MOI.AbstractOptimizer
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

    function Optimizer(verbose, optimtol, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, maxitrefinesteps, alphacorr, predlsmulti, corrlsmulti, itrefinethres)
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


function Optimizer(;
    verbose = false,
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

    return Optimizer(verbose, optimtol, maxiter, predlinesearch, maxpredsmallsteps, maxcorrsteps, corrcheck, maxcorrlsiters, maxitrefinesteps, alphacorr, predlsmulti, corrlsmulti, itrefinethres)
end


const CI = MOI.ConstraintIndex
const VI = MOI.VariableIndex
# TODO maybe don't enforce Float64 type
const SF = Union{MOI.SingleVariable, MOI.ScalarAffineFunction{Float64}, MOI.VectorOfVariables, MOI.VectorAffineFunction{Float64}}
const SS = Union{MOI.EqualTo{Float64}, MOI.GreaterThan{Float64}, MOI.LessThan{Float64}, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives}


MOI.isempty(opt::Optimizer) = (opt.status == :NotLoaded)

function MOI.empty!(opt::Optimizer)
    # TODO empty the data and results
    opt.status == :NotLoaded
end

MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
MOI.supportsconstraint(::Optimizer, ::Type{<:SF}, ::Type{<:SS}) = true

function MOI.copy!(opt::Optimizer, src::MOI.ModelLike; copynames=false, warnattributes=true)
    @assert !copynames

    idxmap = MOIU.IndexMap()

    # model
    # sense_src = MOI.get(src, MOI.ObjectiveSense())
    # mattr_src = MOI.get(src, MOI.ListOfModelAttributesSet())

    # variables
    vn_src = MOI.get(src, MOI.NumberOfVariables())
    vidx_src = MOI.get(src, MOI.ListOfVariableIndices())
    # vattr_src = MOI.get(src, MOI.ListOfVariableAttributesSet())
    j = 0
    for vj in vis_src
        j += 1
        idxmap[vj] = j
    end

    # constraints
    c_src = MOI.get(src, MOI.ListOfConstraints())
    i = 0
    for (F, S) in c_src
        fsc_src = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
        # fsattr_src = MOI.get(src, MOI.ListOfConstraintAttributesSet(fs..))
        for ci in fsc_src
            i += 1
            idxmap[ci] = i
            fi = MOI.get(src, MOI.ConstraintFunction(), ci)
            si = MOI.get(src, MOI.ConstraintSet(), ci)
            # loadconstraint!(dest, fi, si) # TODO build A and cones list
        end
    end

    return idxmap
end




function loaddata!(opt::Optimizer, A::AbstractMatrix{Float64}, b::Vector{Float64}, c::Vector{Float64}, cones::Vector{ConeData}, coneidxs::Vector{AbstractUnitRange})
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




MOI.optimize!(opt::Optimizer) = runalgorithm!(opt)

# function MOI.free!(opt::Optimizer)

# TODO time limit etc
function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    if opt.status in (:Optimal, :NearlyInfeasible, :IllPosed)
        return MOI.Success
    elseif opt.status in (:PredictorFail, :CorrectorFail)
        return MOI.NumericalError
    elseif opt.status == :IterationLimit
        return MOI.IterationLimit
    else
        return MOI.OtherError
    end
end

MOI.get(opt::Optimizer, ::MOI.ObjectiveValue) = opt.pobj

MOI.get(opt::Optimizer, ::MOI.ObjectiveBound) = opt.dobj

function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    # TODO if opt.status in (:Optimal, :NearlyInfeasible, :IllPosed)
    if opt.status == :Optimal
        return 1
    end
end

function MOI.get(opt::Optimizer, ::MOI.PrimalStatus)
    if opt.status == :Optimal
        return MOI.FeasiblePoint
    else
        return MOI.UnknownResultStatus
    end
end

function MOI.get(opt::Optimizer, ::MOI.DualStatus)
    if opt.status == :Optimal
        return MOI.FeasiblePoint
    else
        return MOI.UnknownResultStatus
    end
end



# TODO from ecos test
# const MOIT = MOI.Test
# const MOIB = MOI.Bridges
#
# const MOIU = MOI.Utilities
# MOIU.@model ECOSModelData () (EqualTo, GreaterThan, LessThan) (Zeros, Nonnegatives, Nonpositives, SecondOrderCone, ExponentialCone) () (SingleVariable,) (ScalarAffineFunction,) (VectorOfVariables,) (VectorAffineFunction,)
# const optimizer = MOIU.CachingOptimizer(ECOSModelData{Float64}(), ECOSOptimizer())
#
# # SOC2 requires 1e-4
# const config = MOIT.TestConfig(atol=1e-4, rtol=1e-4)
#
# @testset "Continuous linear problems" begin
#     MOIT.contlineartest(MOIB.SplitInterval{Float64}(optimizer), config)
# end
#
# @testset "Continuous conic problems" begin
#     MOIT.contconictest(MOIB.GeoMean{Float64}(MOIB.RSOC{Float64}(optimizer)), config, ["sdp", "rootdet", "logdet"])
# end
