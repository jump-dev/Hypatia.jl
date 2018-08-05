
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

MOI.isempty(opt::Alfonso.Optimizer) = (opt.status == :NotLoaded)

function MOI.empty!(opt::Alfonso.Optimizer)
    # TODO empty the data and results, or just create a new one?
    opt.status == :NotLoaded
end

MOI.get(::Alfonso.Optimizer, ::MOI.SolverName) = "Alfonso"

MOI.supports(::Alfonso.Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::Alfonso.Optimizer, ::MOI.ObjectiveSense) = true

# TODO don't restrict to Float64 type
const SupportedFuns = Union{
    # MOI.SingleVariable,
    # MOI.ScalarAffineFunction{Float64},
    MOI.VectorOfVariables,
    MOI.VectorAffineFunction{Float64},
    }
const SupportedSets = Union{
    # MOI.EqualTo{Float64},
    # MOI.GreaterThan{Float64},
    # MOI.LessThan{Float64},
    MOI.Zeros,
    MOI.Nonnegatives,
    # MOI.Nonpositives,
    }

MOI.supportsconstraint(::Alfonso.Optimizer, ::Type{<:SupportedFuns}, ::Type{<:SupportedSets}) = true

# build representation as min c'x s.t. Ax = b, x in K
# TODO what if some variables x are in multiple cone constraints?
function MOI.copy!(opt::Alfonso.Optimizer, src::MOI.ModelLike; copynames=false, warnattributes=true)
    @assert !copynames
    idxmap = Dict{MOI.Index,Int}()

    # model
    # mattr_src = MOI.get(src, MOI.ListOfModelAttributesSet())

    # variables
    n = MOI.get(src, MOI.NumberOfVariables()) # columns of A
    # vattr_src = MOI.get(src, MOI.ListOfVariableAttributesSet())
    j = 0
    for vj in MOI.get(src, MOI.ListOfVariableIndices()) # MOI.VariableIndex
        j += 1
        idxmap[vj] = j
    end
    @assert j == n

    # objective function
    (Jc, Vc) = (Int[], Float64[])
    for t in (MOI.get(src, MOI.ObjectiveFunction())).terms
        push!(Jc, idxmap[t.variable_index])
        push!(Vc, t.coefficient)
    end
    ismax = (MOI.get(src, MOI.ObjectiveSense()) == MOI.MaxSense) ? true : false
    if ismax
        Vc .*= -1
    end

    # constraints
    # TODO don't enforce Float64 type
    (IA, JA, VA) = (Int[], Int[], Float64[])
    (Ib, Vb) = (Int[], Float64[])
    cones = Alfonso.ConeData[]
    coneidxs = AbstractUnitRange[]

    i = 0 # MOI constraint objects
    m = 0 # rows of A
    for (F, S) in MOI.get(src, MOI.ListOfConstraints())
        # fsattr_src = MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S})
        for ci in MOI.get(src, MOI.ListOfConstraintIndices{F,S}()) # MOI.ConstraintIndex{F,S}
            # TODO need to access row indices of constraints for get dual on equalities
            i += 1
            idxmap[ci] = i

            fi = MOI.get(src, MOI.ConstraintFunction(), ci)
            si = MOI.get(src, MOI.ConstraintSet(), ci)

            if isa(fi, MOI.AbstractScalarFunction) && isa(si, MOI.AbstractScalarSet)
                error("scalar F and S not yet implemented")
            elseif isa(fi, MOI.AbstractVectorFunction) && isa(si, MOI.AbstractVectorSet)
                dim = MOI.dimension(si)
                @assert MOI.output_dimension(fi) == dim

                if isa(fi, MOI.VectorOfVariables)
                    if isa(si, MOI.Zeros)
                        # variables equal to zero: error
                        error("variables cannot be set equal to zero")
                    else
                        # variables in cone: don't modify A and b
                        push!(cones, buildcone(si)) # TODO implement in barriers.jl
                        push!(coneidxs, [idxmap[vj] for vj in fi.variables])
                    end
                elseif isa(fi, MOI.VectorAffineFunction)
                    # vector affine function: modify A and b
                    append!(Ib, collect(m+1:m+dim))
                    append!(Vb, -fi.constants)

                    for vt in fi.terms
                        push!(IA, m+vt.output_index)
                        push!(JA, idxmap[vt.scalar_term.variable_index])
                        push!(VA, vt.scalar_term.coefficient)
                    end

                    if !isa(si, MOI.Zeros)
                        # add slack variables in new cone
                        append!(IA, collect(m+1:m+dim))
                        append!(JA, collect(n+1:n+dim))
                        append!(VA, fill(-1.0, dim))

                        push!(cones, buildcone(si)) # TODO implement in barriers.jl
                        push!(coneidxs, n+1:n+dim)

                        n += dim
                    end

                    m += dim
                else
                    error("constraint with function $fi in set $si is not supported by Alfonso")
                end
            else
                error("constraint with function $fi in set $si is not supported by Alfonso")
            end
        end
    end

    # TODO maybe should be optional whether to use sparse c and b and A?
    opt.A = dropzeros!(sparse(IA, JA, VA, m, n))
    opt.b = Vector(dropzeros!(sparsevec(Ib, Vb, m)))
    opt.c = Vector(dropzeros!(sparsevec(Jc, Vc, n)))
    opt.cones = cones
    opt.coneidxs = coneidxs
    # opt.ismax = ismax
    opt.status = :Loaded

    return idxmap
end

MOI.optimize!(opt::Alfonso.Optimizer) = runalgorithm!(opt)

# function MOI.free!(opt::Alfonso.Optimizer) # TODO call gc on opt?

function MOI.get(opt::Alfonso.Optimizer, ::MOI.TerminationStatus)
    # TODO time limit etc
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

MOI.get(opt::Alfonso.Optimizer, ::MOI.ObjectiveValue) = opt.pobj
MOI.get(opt::Alfonso.Optimizer, ::MOI.ObjectiveBound) = opt.dobj

function MOI.get(opt::Alfonso.Optimizer, ::MOI.ResultCount)
    # TODO if opt.status in (:Optimal, :NearlyInfeasible, :IllPosed)
    if opt.status == :Optimal
        return 1
    end
end

function MOI.get(opt::Alfonso.Optimizer, ::MOI.PrimalStatus)
    if opt.status == :Optimal
        return MOI.FeasiblePoint
    else
        return MOI.UnknownResultStatus
    end
end

function MOI.get(opt::Alfonso.Optimizer, ::MOI.DualStatus)
    if opt.status == :Optimal
        return MOI.FeasiblePoint
    else
        return MOI.UnknownResultStatus
    end
end
