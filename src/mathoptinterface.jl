
mutable struct Optimizer <: MOI.AbstractOptimizer
    alf::AlfonsoOpt
end

Optimizer() = Optimizer(AlfonsoOpt())

MOI.get(::Optimizer, ::MOI.SolverName) = "Alfonso"

MOI.is_empty(opt::Optimizer) = (get_status(opt.alf) == :NotLoaded)
MOI.empty!(opt::Optimizer) = Optimizer() # TODO empty the data and results, or just create a new one? keep options?

MOI.supports(::Optimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true

# TODO don't restrict to Float64 type
const SupportedFuns = Union{
    MOI.SingleVariable,
    MOI.ScalarAffineFunction{Float64},
    MOI.VectorOfVariables,
    MOI.VectorAffineFunction{Float64},
    }

const SupportedSets = Union{
    MOI.EqualTo{Float64},
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.Nonpositives,
    MOI.SecondOrderCone,
    MOI.RotatedSecondOrderCone,
    MOI.PositiveSemidefiniteConeTriangle,
    MOI.ExponentialCone,
    MOI.PowerCone,
    }

MOI.supports_constraint(::Optimizer, ::Type{<:SupportedFuns}, ::Type{<:SupportedSets}) = true

conefrommoi(s::MOI.Nonnegatives) = NonnegativeCone(MOI.dimension(s))
conefrommoi(s::MOI.Nonpositives) = error("Nonpositive cone not handled yet")
conefrommoi(s::MOI.SecondOrderCone) = SecondOrderCone(MOI.dimension(s))
conefrommoi(s::MOI.RotatedSecondOrderCone) = RotatedSecondOrderCone(MOI.dimension(s))
conefrommoi(s::MOI.PositiveSemidefiniteConeTriangle) = PositiveSemidefiniteCone(MOI.dimension(s))
conefrommoi(s::MOI.ExponentialCone) = ExponentialCone()
conefrommoi(s::MOI.PowerCone) = PowerCone(s.exponent)

# build representation as min c'x s.t. Ax = b, x in K
# TODO what if some variables x are in multiple cone constraints?
function MOI.copy_to(opt::Optimizer, src::MOI.ModelLike; copy_names=false, warn_attributes=true)
    @assert !copy_names
    idxmap = Dict{MOI.Index,MOI.Index}()

    # model
    # mattr_src = MOI.get(src, MOI.ListOfModelAttributesSet())

    # variables
    n = MOI.get(src, MOI.NumberOfVariables()) # columns of A
    # vattr_src = MOI.get(src, MOI.ListOfVariableAttributesSet())
    j = 0
    for vj in MOI.get(src, MOI.ListOfVariableIndices()) # MOI.VariableIndex
        j += 1
        idxmap[vj] = MOI.VariableIndex(j)
    end
    @assert j == n

    # objective function
    (Jc, Vc) = (Int[], Float64[])
    for t in (MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())).terms
        push!(Jc, idxmap[t.variable_index].value)
        push!(Vc, t.coefficient)
    end
    ismax = (MOI.get(src, MOI.ObjectiveSense()) == MOI.MaxSense) ? true : false
    if ismax
        Vc .*= -1
    end

    getsrccons(F, S) = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
    getconfun(conidx) =  MOI.get(src, MOI.ConstraintFunction(), conidx)
    getconset(conidx) =  MOI.get(src, MOI.ConstraintSet(), conidx)

    # pass over vector constraints to construct conic model data
    # TODO don't enforce Float64 type
    (IA, JA, VA) = (Int[], Int[], Float64[])
    (Ib, Vb) = (Int[], Float64[])
    cone = Cone()

    i = 0 # MOI constraint objects
    m = 0 # rows of A

    # equality constraints

    for ci in getsrccons(MOI.SingleVariable, MOI.EqualTo{Float64})
        # TODO can preprocess out maybe
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}(i)

        m += 1
        push!(Ib, m)
        push!(Vb, getconset(ci).value)
        push!(IA, m)
        push!(JA, idxmap[getconfun(ci).variable].value)
        push!(VA, 1.0)
    end

    for ci in getsrccons(MOI.SingleVariable, MOI.Zeros)
        # TODO can preprocess out maybe
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.Zeros}(i)

        m += 1
        push!(IA, m)
        push!(JA, idxmap[getconfun(ci).variable].value)
        push!(VA, 1.0)
    end

    for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(i)

        fi = getconfun(ci)
        m += 1
        push!(Ib, m)
        push!(Vb, getconset(ci).value - fi.constant)
        for vt in fi.terms
            push!(IA, m)
            push!(JA, idxmap[vt.variable_index].value)
            push!(VA, vt.coefficient)
        end
    end

    for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.Zeros}(i)

        fi = getconfun(ci)
        m += 1
        push!(Ib, m)
        push!(Vb, -fi.constant)
        for vt in fi.terms
            push!(IA, m)
            push!(JA, idxmap[vt.variable_index].value)
            push!(VA, vt.coefficient)
        end
    end

    for ci in getsrccons(MOI.VectorOfVariables, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Zeros}(i)

        for vi in getconfun(ci).variables
            m += 1
            push!(IA, m)
            push!(JA, idxmap[vi].value)
            push!(VA, 1.0)
        end
    end

    for ci in getsrccons(MOI.VectorAffineFunction{Float64}, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros}(i)

        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        append!(Ib, collect(m+1:m+dim))
        append!(Vb, -fi.constants)
        for vt in fi.terms
            push!(IA, m + vt.output_index)
            push!(JA, idxmap[vt.scalar_term.variable_index].value)
            push!(VA, vt.scalar_term.coefficient)
        end
        m += dim
    end

    # linear inequality constraints

    nonnegvars = Int[] # for building up a single nonnegative cone

    for S in (MOI.GreaterThan{Float64}, MOI.LessThan{Float64}), ci in getsrccons(MOI.SingleVariable, S)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, S}(i)

        bval = ((S == MOI.GreaterThan{Float64}) ? getconset(ci).lower : getconset(ci).upper)
        if iszero(bval)
            # no need to change A,b
            push!(nonnegvars, idxmap[getconfun(ci).variable].value)
        else
            # add auxiliary variable and equality constraint
            m += 1
            push!(Ib, m)
            push!(Vb, bval)
            push!(IA, m)
            push!(JA, idxmap[getconfun(ci).variable].value)
            push!(VA, 1.0)
            n += 1
            push!(IA, m)
            push!(JA, n)
            push!(VA, ((S == MOI.GreaterThan{Float64}) ? -1.0 : 1.0))
            push!(nonnegvars, n)
        end
    end

    for S in (MOI.GreaterThan{Float64}, MOI.LessThan{Float64}), ci in getsrccons(MOI.ScalarAffineFunction{Float64}, S)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}(i)

        # add auxiliary variable and equality constraint
        fi = getconfun(ci)
        m += 1
        push!(Ib, m)
        push!(Vb, ((S == MOI.GreaterThan{Float64}) ? getconset(ci).lower : getconset(ci).upper) - fi.constant)
        for vt in fi.terms
            push!(IA, m)
            push!(JA, idxmap[vt.variable_index].value)
            push!(VA, vt.coefficient)
        end
        n += 1
        push!(IA, m)
        push!(JA, n)
        push!(VA, ((S == MOI.GreaterThan{Float64}) ? -1.0 : 1.0))
        push!(nonnegvars, n)
    end

    # MOI.VectorOfVariables,
    # MOI.VectorAffineFunction{Float64},
    # MOI.Nonnegatives,
    # MOI.Nonpositives,


    # add single nonnegative cone
    addprimitivecone!(cone, NonnegativeCone(length(nonnegvars)), nonnegvars)


    # TODO iterate over types of S, from easiest to hardest, appending cones in order of toughness
    # then SOC, RSOC, exp, power, sumofsquares, psd

    A = dropzeros!(sparse(IA, JA, VA, m, n))
    b = Vector(dropzeros!(sparsevec(Ib, Vb, m)))
    c = Vector(dropzeros!(sparsevec(Jc, Vc, n)))

    # load problem data through native interface
    load_data!(opt.alf, A, b, c, cone)

    return idxmap
end

MOI.optimize!(opt::Optimizer) = solve!(opt.alf)

# function MOI.free!(opt::Optimizer) # TODO call gc on opt.alf?

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    # TODO time limit etc
    status = get_status(opt.alf)
    if status in (:Optimal, :NearlyInfeasible, :IllPosed)
        return MOI.Success
    elseif status in (:PredictorFail, :CorrectorFail)
        return MOI.NumericalError
    elseif status == :IterationLimit
        return MOI.IterationLimit
    else
        return MOI.OtherError
    end
end

MOI.get(opt::Optimizer, ::MOI.ObjectiveValue) = opt.alf.pobj
MOI.get(opt::Optimizer, ::MOI.ObjectiveBound) = opt.alf.dobj

function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    # TODO if status in (:Optimal, :NearlyInfeasible, :IllPosed)
    status = get_status(opt.alf)
    if status == :Optimal
        return 1
    end
end

function MOI.get(opt::Optimizer, ::MOI.PrimalStatus)
    status = get_status(opt.alf)
    if status == :Optimal
        return MOI.FeasiblePoint
    else
        return MOI.UnknownResultStatus
    end
end

function MOI.get(opt::Optimizer, ::MOI.DualStatus)
    status = get_status(opt.alf)
    if status == :Optimal
        return MOI.FeasiblePoint
    else
        return MOI.UnknownResultStatus
    end
end
