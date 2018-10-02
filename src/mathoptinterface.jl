#=
Copyright 2018, Chris Coey and contributors
=#

mutable struct Optimizer <: MOI.AbstractOptimizer
    alf::HypatiaOpt
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

    function Optimizer(alf::HypatiaOpt)
        opt = new()
        opt.alf = alf
        return opt
    end
end

Optimizer() = Optimizer(HypatiaOpt())

MOI.get(::Optimizer, ::MOI.SolverName) = "Hypatia"

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
    opt.objconst = obj.constant
    opt.objsense = MOI.get(src, MOI.ObjectiveSense())
    opt.c = Vector(sparsevec(Jc, Vc, n))

    # constraints
    getsrccons(F, S) = MOI.get(src, MOI.ListOfConstraintIndices{F,S}())
    getconfun(conidx) = MOI.get(src, MOI.ConstraintFunction(), conidx)
    getconset(conidx) = MOI.get(src, MOI.ConstraintSet(), conidx)
    i = 0 # MOI constraint index

    # equality constraints
    p = 0 # rows of A (equality constraint matrix)
    (IA, JA, VA) = (Int[], Int[], Float64[])
    (Ib, Vb) = (Int[], Float64[])
    (Icpe, Vcpe) = (Int[], Float64[]) # constraint set constants for opt.constrprimeq
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
    opt.A = Matrix(sparse(IA, JA, VA, p, n))
    # opt.A = dropzeros!(sparse(IA, JA, VA, p, n))
    opt.b = Vector(sparsevec(Ib, Vb, p))
    opt.numeqconstrs = i
    opt.constrprimeq = Vector(sparsevec(Icpe, Vcpe, p))
    opt.constroffseteq = constroffseteq

    # conic constraints
    q = 0 # rows of G (cone constraint matrix)
    (IG, JG, VG) = (Int[], Int[], Float64[])
    (Ih, Vh) = (Int[], Float64[])
    (Icpc, Vcpc) = (Int[], Float64[]) # constraint set constants for opt.constrprimeq
    constroffsetcone = Vector{Int}()
    cone = Cone()

    # LP constraints: build up one nonnegative cone and/or one nonpositive cone
    nonnegrows = Int[]
    nonposrows = Int[]

    # TODO also use variable bounds to build one L_inf cone

    for S in (MOI.GreaterThan{Float64}, MOI.LessThan{Float64})
        for ci in getsrccons(MOI.SingleVariable, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.SingleVariable, S}(i)
            push!(constroffsetcone, q)
            q += 1
            push!(IG, q)
            push!(JG, idxmap[getconfun(ci).variable].value)
            push!(VG, -1.0)
            push!(Ih, q)
            if S == MOI.GreaterThan{Float64}
                push!(Vh, -getconset(ci).lower)
                push!(nonnegrows, q)
                push!(Vcpc, getconset(ci).lower)
            else
                push!(Vh, -getconset(ci).upper)
                push!(nonposrows, q)
                push!(Vcpc, getconset(ci).upper)
            end
            push!(Icpc, q)
        end

        for ci in getsrccons(MOI.ScalarAffineFunction{Float64}, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, S}(i)
            push!(constroffsetcone, q)
            q += 1
            fi = getconfun(ci)
            for vt in fi.terms
                push!(IG, q)
                push!(JG, idxmap[vt.variable_index].value)
                push!(VG, -vt.coefficient)
            end
            push!(Ih, q)
            if S == MOI.GreaterThan{Float64}
                push!(Vh, fi.constant - getconset(ci).lower)
                push!(nonnegrows, q)
                push!(Vcpc, getconset(ci).lower)
            else
                push!(Vh, fi.constant - getconset(ci).upper)
                push!(nonposrows, q)
                push!(Vcpc, getconset(ci).upper)
            end
            push!(Icpc, q)
        end
    end

    for S in (MOI.Nonnegatives, MOI.Nonpositives)
        for ci in getsrccons(MOI.VectorOfVariables, S)
            i += 1
            idxmap[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, S}(i)
            push!(constroffsetcone, q)
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
    opt.G = Matrix(sparse(IG, JG, VG, q, n))
    # opt.G = dropzeros!(sparse(IG, JG, VG, q, n))
    opt.h = Vector(sparsevec(Ih, Vh, q))
    opt.cone = cone
    opt.constroffsetcone = constroffsetcone
    opt.constrprimcone = Vector(sparsevec(Icpc, Vcpc, q))

    return idxmap
end

function MOI.optimize!(opt::Optimizer)
    Hypatia.load_data!(opt.alf, opt.c, opt.A, opt.b, opt.G, opt.h, opt.cone) # dense
    Hypatia.solve!(opt.alf)

    opt.x = Hypatia.get_x(opt.alf)
    opt.constrprimeq += opt.b - opt.A*opt.x
    opt.s = Hypatia.get_s(opt.alf)
    opt.constrprimcone += opt.s
    opt.y = Hypatia.get_y(opt.alf)
    opt.z = Hypatia.get_z(opt.alf)

    opt.status = Hypatia.get_status(opt.alf)
    opt.solvetime = Hypatia.get_solvetime(opt.alf)
    opt.pobj = Hypatia.get_pobj(opt.alf)
    opt.dobj = Hypatia.get_dobj(opt.alf)

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
        return opt.pobj + opt.objconst
    elseif opt.objsense == MOI.MaxSense
        return -opt.pobj + opt.objconst
    else
        error("no objective sense is set")
    end
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveBound)
    if opt.objsense == MOI.MinSense
        return opt.dobj + opt.objconst
    elseif opt.objsense == MOI.MaxSense
        return -opt.dobj + opt.objconst
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

function MOI.get(opt::Optimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractScalarSet}
    # scalar set
    i = ci.value
    if i <= opt.numeqconstrs
        # constraint is an equality
        return opt.y[opt.constroffseteq[i]+1]
    else
        # constraint is conic
        i -= opt.numeqconstrs
        return opt.z[opt.constroffsetcone[i]+1]
    end
end
function MOI.get(opt::Optimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractVectorSet}
    # vector set
    i = ci.value
    if i <= opt.numeqconstrs
        # constraint is an equality
        os = opt.constroffseteq
        return opt.y[os[i]+1:os[i+1]]
    else
        # constraint is conic
        i -= opt.numeqconstrs
        os = opt.constroffsetcone
        return opt.z[os[i]+1:os[i+1]]
    end
end
MOI.get(opt::Optimizer, a::MOI.ConstraintDual, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, a, ci)

function MOI.get(opt::Optimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractScalarSet}
    # scalar set
    i = ci.value
    if i <= opt.numeqconstrs
        # constraint is an equality
        return opt.constrprimeq[opt.constroffseteq[i]+1]
    else
        # constraint is conic
        i -= opt.numeqconstrs
        return opt.constrprimcone[opt.constroffsetcone[i]+1]
    end
end
function MOI.get(opt::Optimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractVectorSet}
    # vector set
    i = ci.value
    if i <= opt.numeqconstrs
        # constraint is an equality
        os = opt.constroffseteq
        return opt.constrprimeq[os[i]+1:os[i+1]]
    else
        # constraint is conic
        i -= opt.numeqconstrs
        os = opt.constroffsetcone
        return opt.constrprimcone[os[i]+1:os[i+1]]
    end
end
MOI.get(opt::Optimizer, a::MOI.ConstraintPrimal, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, a, ci)
