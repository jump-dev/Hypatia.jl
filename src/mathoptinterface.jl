
mutable struct Optimizer <: MOI.AbstractOptimizer
    alf::AlfonsoOpt
    objsense::MOI.OptimizationSense

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
    i = 0 # MOI constraint objects

    # equality constraints
    p = 0 # rows of A (equality constraint matrix)
    (IA, JA, VA) = (Int[], Int[], Float64[])
    (Ib, Vb) = (Int[], Float64[])

    for ci in getsrccons(MOI.SingleVariable, MOI.EqualTo{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}(i)

        p += 1
        push!(Ib, p)
        push!(Vb, getconset(ci).value)
        push!(IA, p)
        push!(JA, idxmap[getconfun(ci).variable].value)
        push!(VA, 1.0)
    end

    for ci in getsrccons(MOI.SingleVariable, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.Zeros}(i)

        p += 1
        push!(IA, p)
        push!(JA, idxmap[getconfun(ci).variable].value)
        push!(VA, 1.0)
    end

    for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(i)

        fi = getconfun(ci)
        p += 1
        push!(Ib, p)
        push!(Vb, getconset(ci).value - fi.constant)
        for vt in fi.terms
            push!(IA, p)
            push!(JA, idxmap[vt.variable_index].value)
            push!(VA, vt.coefficient)
        end
    end

    for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.Zeros}(i)

        fi = getconfun(ci)
        p += 1
        push!(Ib, p)
        push!(Vb, -fi.constant)
        for vt in fi.terms
            push!(IA, p)
            push!(JA, idxmap[vt.variable_index].value)
            push!(VA, vt.coefficient)
        end
    end

    for ci in getsrccons(MOI.VectorOfVariables, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Zeros}(i)

        for vi in getconfun(ci).variables
            p += 1
            push!(IA, p)
            push!(JA, idxmap[vi].value)
            push!(VA, 1.0)
        end
    end

    for ci in getsrccons(MOI.VectorAffineFunction{Float64}, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros}(i)

        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        append!(Ib, collect(p+1:p+dim))
        append!(Vb, -fi.constants)
        for vt in fi.terms
            push!(IA, p + vt.output_index)
            push!(JA, idxmap[vt.scalar_term.variable_index].value)
            push!(VA, vt.scalar_term.coefficient)
        end
        p += dim
    end

    # conic constraints
    q = 0 # rows of G (cone constraint matrix)
    (IG, JG, VG) = (Int[], Int[], Float64[])
    (Ih, Vh) = (Int[], Float64[])
    cone = Cone()

    # LP constraints: build up one nonnegative cone and/or one nonpositive cone
    nonnegvars = Int[]
    nonposvars = Int[]
    # TODO also use variable bounds to build one L_inf cone


# TODO use nonpositive cone

    for S in (MOI.GreaterThan{Float64}, MOI.LessThan{Float64})
        for ci in getsrccons(MOI.SingleVariable, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, S}(i)

            bval = ((S == MOI.GreaterThan{Float64}) ? getconset(ci).lower : getconset(ci).upper)
            vn = getconfun(ci).variable
            if iszero(bval)
                # no need to change A,b
                push!(nonnegvars, idxmap[vn].value)
            else
                # add auxiliary variable and equality constraint
                p += 1
                push!(Ib, p)
                push!(Vb, bval)
                push!(IA, p)
                push!(JA, idxmap[vn].value)
                push!(VA, 1.0)
                n += 1
                push!(IA, p)
                push!(JA, n)
                push!(VA, ((S == MOI.GreaterThan{Float64}) ? -1.0 : 1.0))
                push!(nonnegvars, n)
            end
        end

        for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}(i)

            # add auxiliary variable and equality constraint
            fi = getconfun(ci)
            p += 1
            push!(Ib, p)
            push!(Vb, ((S == MOI.GreaterThan{Float64}) ? getconset(ci).lower : getconset(ci).upper) - fi.constant)
            for vt in fi.terms
                push!(IA, p)
                push!(JA, idxmap[vt.variable_index].value)
                push!(VA, vt.coefficient)
            end
            n += 1
            push!(IA, p)
            push!(JA, n)
            push!(VA, ((S == MOI.GreaterThan{Float64}) ? -1.0 : 1.0))
            push!(nonnegvars, n)
        end
    end

    for S in (MOI.Nonnegatives, MOI.Nonpositives)
        for ci in getsrccons(MOI.VectorOfVariables, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, S}(i)

            for vj in getconfun(ci).variables
                col = idxmap[vj].value
                p += 1
                push!(IA, p)
                push!(JA, col)
                push!(VA, ((S == MOI.Nonnegatives) ? 1.0 : -1.0))
                push!(nonnegvars, col)
            end
        end

        for ci in getsrccons(MOI.VectorAffineFunction{Float64}, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, S}(i)

            fi = getconfun(ci)
            dim = MOI.output_dimension(fi)
            for vt in fi.terms
                push!(IA, p + vt.output_index)
                push!(JA, idxmap[vt.scalar_term.variable_index].value)
                push!(VA, vt.scalar_term.coefficient)
            end
            append!(Ib, collect(p+1:p+dim))
            append!(Vb, -fi.constants)
            append!(IA, collect(p+1:p+dim))
            append!(JA, collect(n+1:n+dim))
            append!(VA, fill(((S == MOI.Nonnegatives) ? 1.0 : -1.0), dim))
            p += dim
            append!(nonnegvars, collect(n+1:n+dim))
            n += dim
        end
    end




    # add single nonnegative cone
    addprimitivecone!(cone, NonnegativeCone(length(nonnegvars)), nonnegvars)







    # add other primitive cone constraints (in increasing order of difficulty of evaluating in-cone checks)

    for S in (MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.ExponentialCone, MOI.PowerCone, MOI.PositiveSemidefiniteConeTriangle)
        for ci in getsrccons(MOI.VectorOfVariables, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, S}(i)

            fi = getconfun(ci)
            for vj in fi.variables
                p += 1
                push!(IA, p)
                push!(JA, idxmap[vj].value)
                push!(VA, 1.0)
            end
            addprimitivecone!(cone, conefrommoi(getconset(ci)), [idxmap[vj].value for vj in fi.variables])
        end

        for ci in getsrccons(MOI.VectorAffineFunction{Float64}, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, S}(i)

            fi = getconfun(ci)
            dim = MOI.output_dimension(fi)
            for vt in fi.terms
                push!(IA, p + vt.output_index)
                push!(JA, idxmap[vt.scalar_term.variable_index].value)
                push!(VA, vt.scalar_term.coefficient)
            end
            append!(Ib, collect(p+1:p+dim))
            append!(Vb, -fi.constants)
            append!(IA, collect(p+1:p+dim))
            append!(JA, collect(n+1:n+dim))
            append!(VA, fill(1.0, dim))
            p += dim
            addprimitivecone!(cone, conefrommoi(getconset(ci)), n+1:n+dim)
            n += dim
        end
    end





    # finalize data and load into Alfonso model
    c = Vector(dropzeros!(sparsevec(Jc, Vc, n)))
    A = dropzeros!(sparse(IA, JA, VA, p, n))
    b = Vector(dropzeros!(sparsevec(Ib, Vb, p)))
    G = dropzeros!(sparse(IG, JG, VG, q, n))
    h = Vector(dropzeros!(sparsevec(Ih, Vh, q)))
    Alfonso.load_data!(alf, c, A, b, G, h, cone)

    # set objective sense
    opt.objsense = MOI.get(src, MOI.ObjectiveSense())

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

function MOI.get(opt::Optimizer, ::MOI.ObjectiveValue)
    if opt.objsense == MOI.MinSense
        return get_pobj(opt.alf)
    elseif opt.objsense == MOI.MaxSense
        return -get_pobj(opt.alf)
    else
        return NaN
    end
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveBound)
    if opt.objsense == MOI.MinSense
        return get_dobj(opt.alf)
    elseif opt.objsense == MOI.MaxSense
        return -get_dobj(opt.alf)
    else
        return NaN
    end
end

function MOI.get(opt::Optimizer, ::MOI.ResultCount)
    status = get_status(opt.alf)
    if status in (:Optimal, :NearlyInfeasible, :IllPosed)
        return 1
    end
    return 0
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


# TODO from ECOS:
# Implements getter for result value and statuses
# function MOI.get(instance::Optimizer, ::MOI.TerminationStatus)
#     flag = instance.sol.ret_val
#     if flag == ECOS.ECOS_OPTIMAL
#         MOI.Success
#     elseif flag == ECOS.ECOS_PINF
#         MOI.Success
#     elseif flag == ECOS.ECOS_DINF  # Dual infeasible = primal unbounded, probably
#         MOI.Success
#     elseif flag == ECOS.ECOS_MAXIT
#         MOI.IterationLimit
#     elseif flag == ECOS.ECOS_OPTIMAL + ECOS.ECOS_INACC_OFFSET
#         m.solve_stat = MOI.AlmostSuccess
#     else
#         m.solve_stat = MOI.OtherError
#     end
# end
#
# MOI.get(instance::Optimizer, ::MOI.ObjectiveValue) = instance.sol.objval
# MOI.get(instance::Optimizer, ::MOI.ObjectiveBound) = instance.sol.objbnd
#
# function MOI.get(instance::Optimizer, ::MOI.PrimalStatus)
#     flag = instance.sol.ret_val
#     if flag == ECOS.ECOS_OPTIMAL
#         MOI.FeasiblePoint
#     elseif flag == ECOS.ECOS_PINF
#         MOI.InfeasiblePoint
#     elseif flag == ECOS.ECOS_DINF  # Dual infeasible = primal unbounded, probably
#         MOI.InfeasibilityCertificate
#     elseif flag == ECOS.ECOS_MAXIT
#         MOI.UnknownResultStatus
#     elseif flag == ECOS.ECOS_OPTIMAL + ECOS.ECOS_INACC_OFFSET
#         m.solve_stat = MOI.NearlyFeasiblePoint
#     else
#         m.solve_stat = MOI.OtherResultStatus
#     end
# end
# function MOI.get(instance::Optimizer, ::MOI.DualStatus)
#     flag = instance.sol.ret_val
#     if flag == ECOS.ECOS_OPTIMAL
#         MOI.FeasiblePoint
#     elseif flag == ECOS.ECOS_PINF
#         MOI.InfeasibilityCertificate
#     elseif flag == ECOS.ECOS_DINF  # Dual infeasible = primal unbounded, probably
#         MOI.InfeasiblePoint
#     elseif flag == ECOS.ECOS_MAXIT
#         MOI.UnknownResultStatus
#     elseif flag == ECOS.ECOS_OPTIMAL + ECOS.ECOS_INACC_OFFSET
#         m.solve_stat = MOI.NearlyFeasiblePoint
#     else
#         m.solve_stat = MOI.OtherResultStatus
#     end
# end


MOI.get(opt::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex) = get_x(opt.alf)[vi.value]
MOI.get(opt::Optimizer, a::MOI.VariablePrimal, vi::Vector{MOI.VariableIndex}) = MOI.get.(opt, Ref(a), vi)

# function MOI.get(opt::Optimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{<:MOI.AbstractFunction, MOI.AbstractSet})
# end
# MOI.get(opt::Optimizer, a::MOI.ConstraintPrimal, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, Ref(a), ci)

MOI.get(opt::Optimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex) = get_y(opt.alf)[ci.value]
MOI.get(opt::Optimizer, a::MOI.ConstraintDual, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, Ref(a), ci)




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



#
# get_status(alf::AlfonsoOpt) = alf.status
# get_solvetime(alf::AlfonsoOpt) = alf.solvetime
# get_niters(alf::AlfonsoOpt) = alf.niters
# get_y(alf::AlfonsoOpt) = copy(alf.y)
# get_x(alf::AlfonsoOpt) = copy(alf.x)
# get_tau(alf::AlfonsoOpt) = alf.tau
# get_s(alf::AlfonsoOpt) = copy(alf.s)
# get_kappa(alf::AlfonsoOpt) = alf.kappa
# get_pobj(alf::AlfonsoOpt) = alf.pobj
# get_dobj(alf::AlfonsoOpt) = alf.dobj
# get_dgap(alf::AlfonsoOpt) = alf.dgap
# get_cgap(alf::AlfonsoOpt) = alf.cgap
# get_rel_dgap(alf::AlfonsoOpt) = alf.rel_dgap
# get_rel_cgap(alf::AlfonsoOpt) = alf.rel_cgap
# get_pres(alf::AlfonsoOpt) = copy(alf.pres)
# get_dres(alf::AlfonsoOpt) = copy(alf.dres)
# get_pin(alf::AlfonsoOpt) = alf.pin
# get_din(alf::AlfonsoOpt) = alf.din
# get_rel_pin(alf::AlfonsoOpt) = alf.rel_pin
# get_rel_din(alf::AlfonsoOpt) = alf.rel_din
