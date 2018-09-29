
mutable struct Optimizer <: MOI.AbstractOptimizer
    alf::AlfonsoOpt

    objsense::MOI.OptimizationSense
    constroffsets::Vector{Int}
    numeqconstrs::Int

    x
    s
    y
    z
    status
    solvetime
    pobj
    dobj

    function Optimizer(alf::AlfonsoOpt)
        opt = new()
        opt.alf = alf
        return opt
    end
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
    # MOI.PowerCone,
    }

MOI.supports_constraint(::Optimizer, ::Type{<:SupportedFuns}, ::Type{<:SupportedSets}) = true

conefrommoi(s::MOI.SecondOrderCone) = SecondOrderCone(MOI.dimension(s))
conefrommoi(s::MOI.RotatedSecondOrderCone) = RotatedSecondOrderCone(MOI.dimension(s))
conefrommoi(s::MOI.PositiveSemidefiniteConeTriangle) = PositiveSemidefiniteCone(MOI.dimension(s))
conefrommoi(s::MOI.ExponentialCone) = ExponentialCone()
# conefrommoi(s::MOI.PowerCone) = PowerCone(s.exponent)


# build representation as min c'x s.t. Ax = b, x in K
# TODO what if some variables x are in multiple cone constraints?
function MOI.copy_to(opt::Optimizer, src::MOI.ModelLike; copy_names=false, warn_attributes=true)
    @assert !copy_names
    idxmap = Dict{MOI.Index, MOI.Index}()

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
    getconfun(conidx) = MOI.get(src, MOI.ConstraintFunction(), conidx)
    getconset(conidx) = MOI.get(src, MOI.ConstraintSet(), conidx)

    # pass over vector constraints to construct conic model data
    constroffsets = Vector{Int}()
    i = 0 # MOI constraint objects

    # equality constraints
    p = 0 # rows of A (equality constraint matrix)
    (IA, JA, VA) = (Int[], Int[], Float64[])
    (Ib, Vb) = (Int[], Float64[])

    for S in (MOI.EqualTo{Float64}, MOI.Zeros)
        # TODO can preprocess variables equal to constant
        for ci in getsrccons(MOI.SingleVariable, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, S}(i)
            push!(constroffsets, p)

            p += 1
            push!(IA, p)
            push!(JA, idxmap[getconfun(ci).variable].value)
            push!(VA, 1.0)
            if S == MOI.EqualTo{Float64}
                push!(Ib, p)
                push!(Vb, getconset(ci).value)
            end
        end

        for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}(i)
            push!(constroffsets, p)

            p += 1
            fi = getconfun(ci)
            for vt in fi.terms
                push!(IA, p)
                push!(JA, idxmap[vt.variable_index].value)
                push!(VA, vt.coefficient)
            end
            push!(Ib, p)
            if S == MOI.EqualTo{Float64}
                push!(Vb, getconset(ci).value - fi.constant)
            else
                push!(Vb, -fi.constant)
            end
        end
    end

    # TODO can preprocess variables equal to zero
    for ci in getsrccons(MOI.VectorOfVariables, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Zeros}(i)
        push!(constroffsets, p)

        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        append!(IA, p+1:p+dim)
        append!(JA, idxmap[vi].value for vi in fi.variables)
        append!(VA, ones(dim))
    end

    for ci in getsrccons(MOI.VectorAffineFunction{Float64}, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros}(i)
        push!(constroffsets, p)

        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IA, p + vt.output_index)
            push!(JA, idxmap[vt.scalar_term.variable_index].value)
            push!(VA, vt.scalar_term.coefficient)
        end
        append!(Ib, p+1:p+dim)
        append!(Vb, -fi.constants)
        p += dim
    end

    numeqconstrs = i

    # conic constraints
    q = 0 # rows of G (cone constraint matrix)
    (IG, JG, VG) = (Int[], Int[], Float64[])
    (Ih, Vh) = (Int[], Float64[])
    cone = Cone()

    # LP constraints: build up one nonnegative cone and/or one nonpositive cone
    nonnegrows = Int[]
    nonposrows = Int[]

    # TODO also use variable bounds to build one L_inf cone

    for S in (MOI.GreaterThan{Float64}, MOI.LessThan{Float64})
        for ci in getsrccons(MOI.SingleVariable, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, S}(i)
            push!(constroffsets, q)

            q += 1
            push!(IG, q)
            push!(JG, idxmap[getconfun(ci).variable].value)
            push!(VG, -1.0)
            push!(Ih, q)
            if S == MOI.GreaterThan{Float64}
                push!(Vh, -getconset(ci).lower)
                push!(nonnegrows, q)
            else
                push!(Vh, -getconset(ci).upper)
                push!(nonposrows, q)
            end
        end

        for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}(i)
            push!(constroffsets, q)

            q += 1
            fi = getconfun(ci)
            for vt in fi.terms
                push!(IG, q)
                push!(JG, idxmap[vt.variable_index].value)
                push!(VG, -vt.coefficient)
            end
            push!(Ih, q)
            if S == MOI.GreaterThan{Float64}
                push!(Vh, -getconset(ci).lower + fi.constant)
                push!(nonnegrows, q)
            else
                push!(Vh, -getconset(ci).upper + fi.constant)
                push!(nonposrows, q)
            end
        end
    end

    for S in (MOI.Nonnegatives, MOI.Nonpositives)
        for ci in getsrccons(MOI.VectorOfVariables, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, S}(i)
            push!(constroffsets, q)

            for vj in getconfun(ci).variables
                q += 1
                push!(IG, q)
                push!(JG, idxmap[vj].value)
                push!(VG, -1.0)
                if S == MOI.Nonnegatives
                    push!(nonnegrows, q)
                else
                    push!(nonposrows, q)
                end
            end
        end

        for ci in getsrccons(MOI.VectorAffineFunction{Float64}, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, S}(i)
            push!(constroffsets, q)

            fi = getconfun(ci)
            dim = MOI.output_dimension(fi)
            for vt in fi.terms
                push!(IG, q + vt.output_index)
                push!(JG, idxmap[vt.scalar_term.variable_index].value)
                push!(VG, -vt.scalar_term.coefficient)
            end
            append!(Ih, q+1:q+dim)
            append!(Vh, fi.constants)
            if S == MOI.Nonnegatives
                append!(nonnegrows, q+1:q+dim)
            else
                append!(nonposrows, q+1:q+dim)
            end
            q += dim
        end
    end

    addprimitivecone!(cone, NonnegativeCone(length(nonnegrows)), nonnegrows)
    addprimitivecone!(cone, NonpositiveCone(length(nonposrows)), nonposrows)

    # non-LP conic constraints

    for S in (MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.ExponentialCone, MOI.PowerCone, MOI.PositiveSemidefiniteConeTriangle)
        for ci in getsrccons(MOI.VectorOfVariables, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, S}(i)
            push!(constroffsets, q)

            fi = getconfun(ci)
            dim = MOI.output_dimension(fi)
            append!(IG, q+1:q+dim)
            append!(JG, idxmap[vj].value for vj in fi.variables)
            append!(VG, -ones(dim))
            addprimitivecone!(cone, conefrommoi(getconset(ci)), q+1:q+dim)
            q += dim
        end

        for ci in getsrccons(MOI.VectorAffineFunction{Float64}, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, S}(i)
            push!(constroffsets, q)

            fi = getconfun(ci)
            dim = MOI.output_dimension(fi)
            for vt in fi.terms
                push!(IG, q + vt.output_index)
                push!(JG, idxmap[vt.scalar_term.variable_index].value)
                push!(VG, -vt.scalar_term.coefficient)
            end
            append!(Ih, q+1:q+dim)
            append!(Vh, fi.constants)
            addprimitivecone!(cone, conefrommoi(getconset(ci)), q+1:q+dim)
            q += dim
        end
    end

    # finalize data and load into Alfonso model
    c = Vector(dropzeros!(sparsevec(Jc, Vc, n)))
    A = dropzeros!(sparse(IA, JA, VA, p, n))
    b = Vector(dropzeros!(sparsevec(Ib, Vb, p)))
    G = dropzeros!(sparse(IG, JG, VG, q, n))
    h = Vector(dropzeros!(sparsevec(Ih, Vh, q)))
    # TODO make converting to dense optional
    # Alfonso.load_data!(opt.alf, c, A, b, G, h, cone) # sparse
    Alfonso.load_data!(opt.alf, c, Matrix(A), b, Matrix(G), h, cone) # dense

    # store information needed by MOI getter functions
    opt.objsense = MOI.get(src, MOI.ObjectiveSense())
    opt.constroffsets = constroffsets
    opt.numeqconstrs = numeqconstrs

    return idxmap
end

function MOI.optimize!(opt::Optimizer)
    solve!(opt.alf)

    opt.x = get_x(opt.alf)
    opt.s = get_s(opt.alf)
    opt.y = get_y(opt.alf)
    opt.z = get_z(opt.alf)
    opt.status = get_status(opt.alf)
    opt.solvetime = get_solvetime(opt.alf)
    opt.pobj = get_pobj(opt.alf)
    opt.dobj = get_dobj(opt.alf)

    return nothing
end

# function MOI.free!(opt::Optimizer) # TODO call gc on opt.alf?

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    # TODO time limit etc
    if opt.status in (:Optimal, :PrimalInfeasible, :DualInfeasible, :IllPosed)
        return MOI.Success
    elseif opt.status in (:PredictorFail, :CorrectorFail)
        return MOI.NumericalError
    elseif opt.status == :IterationLimit
        return MOI.IterationLimit
    else
        return MOI.OtherError
    end
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveValue)
    if opt.objsense == MOI.MinSense
        return opt.pobj
    elseif opt.objsense == MOI.MaxSense
        return -opt.pobj
    else
        error("no objective sense is set")
    end
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveBound)
    if opt.objsense == MOI.MinSense
        return opt.dobj
    elseif opt.objsense == MOI.MaxSense
        return -opt.dobj
    else
        error("no objective sense is set")
    end
end

function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    if opt.status in (:Optimal, :PrimalInfeasible, :DualInfeasible, :IllPosed)
        return 1
    end
    return 0
end

function MOI.get(opt::Optimizer, ::MOI.PrimalStatus)
    if opt.status == :Optimal
        return MOI.FeasiblePoint
    elseif opt.status == :PrimalInfeasible
        return MOI.InfeasiblePoint
    elseif opt.status == :DualInfeasible
        return MOI.InfeasibilityCertificate
    elseif opt.status == :IllPosed
        return MOI.UnknownResultStatus # TODO later distinguish primal/dual ill posed certificates
    else
        return MOI.UnknownResultStatus
    end
end

function MOI.get(opt::Optimizer, ::MOI.DualStatus)
    if opt.status == :Optimal
        return MOI.FeasiblePoint
    elseif opt.status == :PrimalInfeasible
        return MOI.InfeasibilityCertificate
    elseif opt.status == :DualInfeasible
        return MOI.InfeasiblePoint
    elseif opt.status == :IllPosed
        return MOI.UnknownResultStatus # TODO later distinguish primal/dual ill posed certificates
    else
        return MOI.UnknownResultStatus
    end
end

MOI.get(opt::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex) = opt.x[vi.value]
MOI.get(opt::Optimizer, a::MOI.VariablePrimal, vi::Vector{MOI.VariableIndex}) = MOI.get.(opt, a, vi)

function MOI.get(opt::Optimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F, S}
    os = opt.constroffsets
    i = ci.value
    if i <= opt.numeqconstrs
        # constraint is an equality
        return opt.y[os[i]+1:os[i+1]]
    else
        # constraint is conic
        return opt.z[os[i]+1:os[i+1]]
    end
end
MOI.get(opt::Optimizer, a::MOI.ConstraintDual, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, a, ci)

# TODO use x and s. if conic then just return s. otherwise have to use x - rhs after saving rhs in opt maybe
MOI.get(opt::Optimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex) = NaN
MOI.get(opt::Optimizer, a::MOI.ConstraintPrimal, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, a, ci)





# function MOI.get(instance::Optimizer, ::MOI.ConstraintPrimal, ci::CI{<:MOI.AbstractFunction, MOI.Zeros})
#     rows = constrrows(instance, ci)
#     zeros(length(rows))
# end
# function MOI.get(instance::Optimizer, ::MOI.ConstraintPrimal, ci::CI{<:MOI.AbstractFunction, <:MOI.EqualTo})
#     offset = constroffset(instance, ci)
#     setconstant(instance, offset, ci)
# end
# function MOI.get(instance::Optimizer, ::MOI.ConstraintPrimal, ci::CI{<:MOI.AbstractFunction, S}) where S <: MOI.AbstractSet
#     offset = constroffset(instance, ci)
#     rows = constrrows(instance, ci)
#     _unshift(instance, offset, scalecoef(rows, reorderval(instance.sol.slack[offset .+ rows], S), false, S), ci)
# end
#
# _dual(instance, ci::CI{<:MOI.AbstractFunction, <:ZeroCones}) = instance.sol.dual_eq
# _dual(instance, ci::CI) = instance.sol.dual_ineq
# function MOI.get(instance::Optimizer, ::MOI.ConstraintDual, ci::CI{<:MOI.AbstractFunction, S}) where S <: MOI.AbstractSet
#     offset = constroffset(instance, ci)
#     rows = constrrows(instance, ci)
#     scalecoef(rows, reorderval(_dual(instance, ci)[offset .+ rows], S), false, S)
# end
#
# MOI.get(instance::Optimizer, ::MOI.ResultCount) = 1
