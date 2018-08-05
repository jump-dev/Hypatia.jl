
import MathOptInterface
const MOI = MathOptInterface
const MOIU = MOI.Utilities


mutable struct Optimizer <: MOI.AbstractOptimizer
    alf::AlfonsoOpt
end

Optimizer() = Optimizer(AlfonsoOpt())

MOI.isempty(opt::Alfonso.Optimizer) = (opt.alf.status == :NotLoaded)

function MOI.empty!(opt::Alfonso.Optimizer)
    # TODO empty the data and results, or just create a new one?
    opt.alf.status == :NotLoaded
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

    A = dropzeros!(sparse(IA, JA, VA, m, n))
    b = Vector(dropzeros!(sparsevec(Ib, Vb, m)))
    c = Vector(dropzeros!(sparsevec(Jc, Vc, n)))

    loaddata!(opt.alf, A, b, c, cones, coneidxs)

    return idxmap
end

function MOI.optimize!(opt::Alfonso.Optimizer)
    runalgorithm!(opt.alf)
    # TODO do things to set MOI-related fields in opt
end

# function MOI.free!(opt::Alfonso.Optimizer) # TODO call gc on opt.alf?

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
