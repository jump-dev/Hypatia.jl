#=
Copyright 2018, Chris Coey and contributors
=#

mutable struct HypatiaOptimizer <: MOI.AbstractOptimizer
    opt::Optimizer
    usedense::Bool
    c::Vector{Float64}          # linear cost vector, size n
    A::AbstractMatrix{Float64}  # equality constraint matrix, size p*n
    b::Vector{Float64}          # equality constraint vector, size p
    G::AbstractMatrix{Float64}  # cone constraint matrix, size q*n
    h::Vector{Float64}          # cone constraint vector, size q
    cone::Cone                  # primal constraint cone object
    objsense::MOI.OptimizationSense
    objconst::Float64
    numeqconstrs::Int
    constroffseteq::Vector{Int}
    constrprimeq::Vector{Float64}
    constroffsetcone::Vector{Int}
    constrprimcone::Vector{Float64}
    x::Vector{Float64}
    s::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    status::Symbol
    solvetime::Float64
    pobj::Float64
    dobj::Float64

    function HypatiaOptimizer(opt::Optimizer, usedense::Bool)
        moiopt = new()
        moiopt.opt = opt
        moiopt.usedense = usedense
        return moiopt
    end
end

HypatiaOptimizer(; usedense::Bool=true) = HypatiaOptimizer(Optimizer(), usedense)

MOI.get(::HypatiaOptimizer, ::MOI.SolverName) = "Hypatia"

MOI.is_empty(moiopt::HypatiaOptimizer) = (get_status(moiopt.opt) == :NotLoaded)
MOI.empty!(moiopt::HypatiaOptimizer) = HypatiaOptimizer() # TODO empty the data and results, or just create a new one? keep options?

MOI.supports(::HypatiaOptimizer, ::MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}) = true
MOI.supports(::HypatiaOptimizer, ::MOI.ObjectiveSense) = true

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

MOI.supports_constraint(::HypatiaOptimizer, ::Type{<:SupportedFuns}, ::Type{<:SupportedSets}) = true

conefrommoi(s::MOI.SecondOrderCone) = SecondOrderCone(MOI.dimension(s))
conefrommoi(s::MOI.RotatedSecondOrderCone) = RotatedSecondOrderCone(MOI.dimension(s))
conefrommoi(s::MOI.PositiveSemidefiniteConeTriangle) = PositiveSemidefiniteCone(MOI.dimension(s))
conefrommoi(s::MOI.ExponentialCone) = ExponentialCone()
# conefrommoi(s::MOI.PowerCone) = PowerCone(s.exponent)

# build representation as min c'x s.t. Ax = b, x in K
# TODO what if some variables x are in multiple cone constraints?
function MOI.copy_to(moiopt::HypatiaOptimizer, src::MOI.ModelLike; copy_names=false, warn_attributes=true)
    @assert !copy_names
    idxmap = Dict{MOI.Index, MOI.Index}()

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
    obj = MOI.get(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    (Jc, Vc) = (Int[], Float64[])
    for t in obj.terms
        push!(Jc, idxmap[t.variable_index].value)
        push!(Vc, t.coefficient)
    end
    if MOI.get(src, MOI.ObjectiveSense()) == MOI.MaxSense
        Vc .*= -1.0
    end
    moiopt.objconst = obj.constant
    moiopt.objsense = MOI.get(src, MOI.ObjectiveSense())
    moiopt.c = Vector(sparsevec(Jc, Vc, n))

    # constraints
    getsrccons(F, S) = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
    getconfun(conidx) = MOI.get(src, MOI.ConstraintFunction(), conidx)
    getconset(conidx) = MOI.get(src, MOI.ConstraintSet(), conidx)
    i = 0 # MOI constraint index

    # equality constraints
    p = 0 # rows of A (equality constraint matrix)
    (IA, JA, VA) = (Int[], Int[], Float64[])
    (Ib, Vb) = (Int[], Float64[])
    (Icpe, Vcpe) = (Int[], Float64[]) # constraint set constants for moiopt.constrprimeq
    constroffseteq = Vector{Int}()

    for S in (MOI.EqualTo{Float64}, MOI.Zeros)
        # TODO can preprocess variables equal to constant
        for ci in getsrccons(MOI.SingleVariable, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, S}(i)
            push!(constroffseteq, p)
            p += 1
            push!(IA, p)
            push!(JA, idxmap[getconfun(ci).variable].value)
            push!(VA, -1.0)
            if S == MOI.EqualTo{Float64}
                push!(Ib, p)
                push!(Vb, -getconset(ci).value)
                push!(Icpe, p)
                push!(Vcpe, getconset(ci).value)
            end
        end

        for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}(i)
            push!(constroffseteq, p)
            p += 1
            fi = getconfun(ci)
            for vt in fi.terms
                push!(IA, p)
                push!(JA, idxmap[vt.variable_index].value)
                push!(VA, -vt.coefficient)
            end
            push!(Ib, p)
            if S == MOI.EqualTo{Float64}
                push!(Vb, fi.constant - getconset(ci).value)
                push!(Icpe, p)
                push!(Vcpe, getconset(ci).value)
            else
                push!(Vb, fi.constant)
            end
        end
    end

    # TODO can preprocess variables equal to zero
    for ci in getsrccons(MOI.VectorOfVariables, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Zeros}(i)
        push!(constroffseteq, p)
        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        append!(IA, p+1:p+dim)
        append!(JA, idxmap[vi].value for vi in fi.variables)
        append!(VA, -ones(dim))
    end

    for ci in getsrccons(MOI.VectorAffineFunction{Float64}, MOI.Zeros)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros}(i)
        push!(constroffseteq, p)
        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IA, p + vt.output_index)
            push!(JA, idxmap[vt.scalar_term.variable_index].value)
            push!(VA, -vt.scalar_term.coefficient)
        end
        append!(Ib, p+1:p+dim)
        append!(Vb, fi.constants)
        p += dim
    end

    push!(constroffseteq, p)
    if moiopt.usedense
        moiopt.A = Matrix(sparse(IA, JA, VA, p, n))
    else
        moiopt.A = dropzeros!(sparse(IA, JA, VA, p, n))
    end
    moiopt.b = Vector(sparsevec(Ib, Vb, p)) # TODO if type less strongly, this can be sparse too
    moiopt.numeqconstrs = i
    moiopt.constrprimeq = Vector(sparsevec(Icpe, Vcpe, p))
    moiopt.constroffseteq = constroffseteq

    # conic constraints
    q = 0 # rows of G (cone constraint matrix)
    (IG, JG, VG) = (Int[], Int[], Float64[])
    (Ih, Vh) = (Int[], Float64[])
    (Icpc, Vcpc) = (Int[], Float64[]) # constraint set constants for moiopt.constrprimeq
    constroffsetcone = Vector{Int}()
    cone = Cone()

    # build up one nonnegative cone
    nonnegstart = q

    for ci in getsrccons(MOI.SingleVariable, MOI.GreaterThan{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(i)
        push!(constroffsetcone, q)
        q += 1
        push!(IG, q)
        push!(JG, idxmap[getconfun(ci).variable].value)
        push!(VG, -1.0)
        push!(Ih, q)
        push!(Vh, -getconset(ci).lower)
        push!(Vcpc, getconset(ci).lower)
        push!(Icpc, q)
    end

    for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}(i)
        push!(constroffsetcone, q)
        q += 1
        fi = getconfun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idxmap[vt.variable_index].value)
            push!(VG, -vt.coefficient)
        end
        push!(Ih, q)
        push!(Vh, fi.constant - getconset(ci).lower)
        push!(Vcpc, getconset(ci).lower)
        push!(Icpc, q)
    end

    for ci in getsrccons(MOI.VectorOfVariables, MOI.Nonnegatives)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonnegatives}(i)
        push!(constroffsetcone, q)
        for vj in getconfun(ci).variables
            q += 1
            push!(IG, q)
            push!(JG, idxmap[vj].value)
            push!(VG, -1.0)
        end
    end

    for ci in getsrccons(MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives}(i)
        push!(constroffsetcone, q)
        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IG, q + vt.output_index)
            push!(JG, idxmap[vt.scalar_term.variable_index].value)
            push!(VG, -vt.scalar_term.coefficient)
        end
        append!(Ih, q+1:q+dim)
        append!(Vh, fi.constants)
        q += dim
    end

    addprimitivecone!(cone, NonnegativeCone(q - nonnegstart), nonnegstart+1:q)

    # build up one nonpositive cone
    nonposstart = q

    for ci in getsrccons(MOI.SingleVariable, MOI.LessThan{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(i)
        push!(constroffsetcone, q)
        q += 1
        push!(IG, q)
        push!(JG, idxmap[getconfun(ci).variable].value)
        push!(VG, -1.0)
        push!(Ih, q)
        push!(Vh, -getconset(ci).upper)
        push!(Vcpc, getconset(ci).upper)
        push!(Icpc, q)
    end

    for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64})
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(i)
        push!(constroffsetcone, q)
        q += 1
        fi = getconfun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idxmap[vt.variable_index].value)
            push!(VG, -vt.coefficient)
        end
        push!(Ih, q)
        push!(Vh, fi.constant - getconset(ci).upper)
        push!(Vcpc, getconset(ci).upper)
        push!(Icpc, q)
    end

    for ci in getsrccons(MOI.VectorOfVariables, MOI.Nonpositives)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonpositives}(i)
        push!(constroffsetcone, q)
        for vj in getconfun(ci).variables
            q += 1
            push!(IG, q)
            push!(JG, idxmap[vj].value)
            push!(VG, -1.0)
        end
    end

    for ci in getsrccons(MOI.VectorAffineFunction{Float64}, MOI.Nonpositives)
        i += 1
        idxmap[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonpositives}(i)
        push!(constroffsetcone, q)
        fi = getconfun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IG, q + vt.output_index)
            push!(JG, idxmap[vt.scalar_term.variable_index].value)
            push!(VG, -vt.scalar_term.coefficient)
        end
        append!(Ih, q+1:q+dim)
        append!(Vh, fi.constants)
        q += dim
    end

    addprimitivecone!(cone, NonpositiveCone(q - nonposstart), nonposstart+1:q)

    # TODO also use variable bounds to build one L_inf cone

    # non-LP conic constraints

    for S in (MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.ExponentialCone, MOI.PowerCone, MOI.PositiveSemidefiniteConeTriangle)
        for ci in getsrccons(MOI.VectorOfVariables, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, S}(i)
            push!(constroffsetcone, q)
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
            push!(constroffsetcone, q)
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

    push!(constroffsetcone, q)
    if moiopt.usedense
        moiopt.G = Matrix(sparse(IG, JG, VG, q, n))
    else
        moiopt.G = dropzeros!(sparse(IG, JG, VG, q, n))
    end
    moiopt.h = Vector(sparsevec(Ih, Vh, q)) # TODO if type less strongly, this can be sparse too
    moiopt.cone = cone
    moiopt.constroffsetcone = constroffsetcone
    moiopt.constrprimcone = Vector(sparsevec(Icpc, Vcpc, q))

    return idxmap
end

function MOI.optimize!(moiopt::HypatiaOptimizer)
    opt = moiopt.opt
    load_data!(opt, moiopt.c, moiopt.A, moiopt.b, moiopt.G, moiopt.h, moiopt.cone) # dense
    solve!(opt)

    moiopt.x = get_x(opt)
    moiopt.constrprimeq += moiopt.b - moiopt.A*moiopt.x
    moiopt.s = get_s(opt)
    moiopt.constrprimcone += moiopt.s
    moiopt.y = get_y(opt)
    moiopt.z = get_z(opt)

    moiopt.status = get_status(opt)
    moiopt.solvetime = get_solvetime(opt)
    moiopt.pobj = get_pobj(opt)
    moiopt.dobj = get_dobj(opt)

    return nothing
end

# function MOI.free!(moiopt::HypatiaOptimizer) # TODO call gc on moiopt.opt?

function MOI.get(moiopt::HypatiaOptimizer, ::MOI.TerminationStatus)
    # TODO time limit etc
    if moiopt.status in (:Optimal, :PrimalInfeasible, :DualInfeasible, :IllPosed)
        return MOI.Success
    elseif moiopt.status in (:PredictorFail, :CorrectorFail)
        return MOI.NumericalError
    elseif moiopt.status == :IterationLimit
        return MOI.IterationLimit
    else
        return MOI.OtherError
    end
end

function MOI.get(moiopt::HypatiaOptimizer, ::MOI.ObjectiveValue)
    if moiopt.objsense == MOI.MinSense
        return moiopt.pobj + moiopt.objconst
    elseif moiopt.objsense == MOI.MaxSense
        return -moiopt.pobj + moiopt.objconst
    else
        error("no objective sense is set")
    end
end

function MOI.get(moiopt::HypatiaOptimizer, ::MOI.ObjectiveBound)
    if moiopt.objsense == MOI.MinSense
        return moiopt.dobj + moiopt.objconst
    elseif moiopt.objsense == MOI.MaxSense
        return -moiopt.dobj + moiopt.objconst
    else
        error("no objective sense is set")
    end
end

function MOI.get(moiopt::HypatiaOptimizer, ::MOI.ResultCount)
    if moiopt.status in (:Optimal, :PrimalInfeasible, :DualInfeasible, :IllPosed)
        return 1
    end
    return 0
end

function MOI.get(moiopt::HypatiaOptimizer, ::MOI.PrimalStatus)
    if moiopt.status == :Optimal
        return MOI.FeasiblePoint
    elseif moiopt.status == :PrimalInfeasible
        return MOI.InfeasiblePoint
    elseif moiopt.status == :DualInfeasible
        return MOI.InfeasibilityCertificate
    elseif moiopt.status == :IllPosed
        return MOI.UnknownResultStatus # TODO later distinguish primal/dual ill posed certificates
    else
        return MOI.UnknownResultStatus
    end
end

function MOI.get(moiopt::HypatiaOptimizer, ::MOI.DualStatus)
    if moiopt.status == :Optimal
        return MOI.FeasiblePoint
    elseif moiopt.status == :PrimalInfeasible
        return MOI.InfeasibilityCertificate
    elseif moiopt.status == :DualInfeasible
        return MOI.InfeasiblePoint
    elseif moiopt.status == :IllPosed
        return MOI.UnknownResultStatus # TODO later distinguish primal/dual ill posed certificates
    else
        return MOI.UnknownResultStatus
    end
end

MOI.get(moiopt::HypatiaOptimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex) = moiopt.x[vi.value]
MOI.get(moiopt::HypatiaOptimizer, a::MOI.VariablePrimal, vi::Vector{MOI.VariableIndex}) = MOI.get.(moiopt, a, vi)

function MOI.get(moiopt::HypatiaOptimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractScalarSet}
    # scalar set
    i = ci.value
    if i <= moiopt.numeqconstrs
        # constraint is an equality
        return moiopt.y[moiopt.constroffseteq[i]+1]
    else
        # constraint is conic
        i -= moiopt.numeqconstrs
        return moiopt.z[moiopt.constroffsetcone[i]+1]
    end
end
function MOI.get(moiopt::HypatiaOptimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractVectorSet}
    # vector set
    i = ci.value
    if i <= moiopt.numeqconstrs
        # constraint is an equality
        os = moiopt.constroffseteq
        return moiopt.y[os[i]+1:os[i+1]]
    else
        # constraint is conic
        i -= moiopt.numeqconstrs
        os = moiopt.constroffsetcone
        return moiopt.z[os[i]+1:os[i+1]]
    end
end
MOI.get(moiopt::HypatiaOptimizer, a::MOI.ConstraintDual, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(moiopt, a, ci)

function MOI.get(moiopt::HypatiaOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractScalarSet}
    # scalar set
    i = ci.value
    if i <= moiopt.numeqconstrs
        # constraint is an equality
        return moiopt.constrprimeq[moiopt.constroffseteq[i]+1]
    else
        # constraint is conic
        i -= moiopt.numeqconstrs
        return moiopt.constrprimcone[moiopt.constroffsetcone[i]+1]
    end
end
function MOI.get(moiopt::HypatiaOptimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractVectorSet}
    # vector set
    i = ci.value
    if i <= moiopt.numeqconstrs
        # constraint is an equality
        os = moiopt.constroffseteq
        return moiopt.constrprimeq[os[i]+1:os[i+1]]
    else
        # constraint is conic
        i -= moiopt.numeqconstrs
        os = moiopt.constroffsetcone
        return moiopt.constrprimcone[os[i]+1:os[i+1]]
    end
end
MOI.get(moiopt::HypatiaOptimizer, a::MOI.ConstraintPrimal, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(moiopt, a, ci)
