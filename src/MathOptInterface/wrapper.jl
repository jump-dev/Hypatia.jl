#=
MathOptInterface wrapper of Hypatia solver
=#

const VI = MOI.VariableIndex
const SAF = MOI.ScalarAffineFunction
const VV = MOI.VectorOfVariables
const VAF = MOI.VectorAffineFunction

export Optimizer

"""
$(TYPEDEF)

A MathOptInterface optimizer type for Hypatia.
"""
mutable struct Optimizer{T <: Real} <: MOI.AbstractOptimizer
    solver::Solvers.Solver{T} # Hypatia solver object
    model::Models.Model{T} # Hypatia model object

    # result data
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}

    # data for transforming certificates
    obj_sense::MOI.OptimizationSense
    zeros_idxs::Vector{UnitRange{Int}}
    zeros_primal::Vector{T}
    cones_idxs::Vector{UnitRange{Int}}
    other_cones_start::Int
    other_cones::Vector{MOI.AbstractVectorSet}

    function Optimizer{T}(; solver_options...) where {T <: Real}
        opt = new{T}()
        opt.solver = Solvers.Solver{T}(; solver_options...)
        return opt
    end
end

Optimizer(; options...) = Optimizer{Float64}(; options...) # default to Float64

MOI.get(::Optimizer, ::MOI.SolverName) = "Hypatia"

MOI.get(opt::Optimizer, ::MOI.RawSolver) = opt.solver

MOI.is_empty(opt::Optimizer) = (opt.solver.status == Solvers.NotLoaded)

MOI.empty!(opt::Optimizer) = (opt.solver.status = Solvers.NotLoaded)

MOI.supports(
    ::Optimizer{T},
    ::Union{MOI.ObjectiveSense, MOI.ObjectiveFunction{SAF{T}}},
    ) where {T <: Real} = true

MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{<:Union{VV, VAF{T}}},
    ::Type{<:SupportedCones{T}},
    ) where {T <: Real} = true

# build representation as min c'x s.t. A*x = b, h - G*x in K
function MOI.copy_to(
    opt::Optimizer{T},
    src::MOI.ModelLike,
    ) where {T <: Real}
    idx_map = MOI.Utilities.IndexMap()

    # variables
    n = MOI.get(src, MOI.NumberOfVariables()) # columns of A
    j = 0
    for vj in MOI.get(src, MOI.ListOfVariableIndices())
        j += 1
        idx_map[vj] = VI(j)
    end
    @assert j == n
    for attr in MOI.get(src, MOI.ListOfVariableAttributesSet())
        if attr == MOI.VariableName() || attr == MOI.VariablePrimalStart()
            continue
        end
        throw(MOI.UnsupportedAttribute(attr))
    end

    # objective
    opt.obj_sense = MOI.MIN_SENSE
    (Jc, Vc) = (Int[], T[])
    obj_offset = 0.0
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        if attr == MOI.Name()
            continue
        elseif attr == MOI.ObjectiveSense()
            opt.obj_sense = MOI.get(src, MOI.ObjectiveSense())
        elseif attr isa MOI.ObjectiveFunction
            F = MOI.get(src, MOI.ObjectiveFunctionType())
            if F != SAF{T}
                error("function type $F not supported")
            end
            obj = MOI.get(src, MOI.ObjectiveFunction{F}())
            append!(Jc, (idx_map[t.variable].value for t in obj.terms))
            append!(Vc, (t.coefficient for t in obj.terms))
            obj_offset = obj.constant
        else
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    if opt.obj_sense == MOI.MAX_SENSE
        Vc .*= -1
        obj_offset *= -1
    end
    model_c = Vector(sparsevec(Jc, Vc, n))

    # constraints
    get_src_cons(F, S) = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    get_con_fun(con_idx) = MOI.get(src, MOI.ConstraintFunction(), con_idx)
    get_con_set(con_idx) = MOI.get(src, MOI.ConstraintSet(), con_idx)

    # equality constraints
    i = 1 # MOI constraint index
    p = 0 # rows of A (equality constraint matrix)
    (IA, JA, VA) = (Int[], Int[], T[])
    (Ib, Vb) = (Int[], T[])
    zeros_idxs = Vector{UnitRange{Int}}()

    for F in (VV, VAF{T}), ci in get_src_cons(F, MOI.Zeros)
        idx_map[ci] = MOI.ConstraintIndex{F, MOI.Zeros}(i)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        if F == VV
            append!(IA, (p + 1):(p + dim))
            append!(JA, idx_map[vi].value for vi in fi.variables)
            append!(VA, -ones(T, dim))
        else
            for vt in fi.terms
                push!(IA, p + vt.output_index)
                push!(JA, idx_map[vt.scalar_term.variable].value)
                push!(VA, -vt.scalar_term.coefficient)
            end
            append!(Ib, (p + 1):(p + dim))
            append!(Vb, fi.constants)
        end
        push!(zeros_idxs, p .+ (1:dim))
        p += dim
        i += 1
    end

    model_A = dropzeros!(sparse(IA, JA, VA, p, n))
    model_b = Vector(sparsevec(Ib, Vb, p))
    opt.zeros_idxs = zeros_idxs

    # conic constraints
    other_cones = MOI.AbstractVectorSet[]
    i = 1 # MOI constraint index
    q = 0 # rows of G (cone constraint matrix)
    (IG, JG, VG) = (Int[], Int[], T[])
    (Ih, Vh) = (Int[], T[])
    cones_idxs = Vector{UnitRange{Int}}()
    cones = Cones.Cone{T}[]

    # build up one nonnegative cone
    for F in (VV, VAF{T}), ci in get_src_cons(F, MOI.Nonnegatives)
        idx_map[ci] = MOI.ConstraintIndex{F, MOI.Nonnegatives}(i)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        idxs = q .+ (1:dim)
        if F == VV
            append!(IG, idxs)
            append!(JG, (idx_map[vj].value for vj in fi.variables))
            append!(VG, fill(-one(T), dim))
        else
            for vt in fi.terms
                push!(IG, q + vt.output_index)
                push!(JG, idx_map[vt.scalar_term.variable].value)
                push!(VG, -vt.scalar_term.coefficient)
            end
            append!(Ih, idxs)
            append!(Vh, fi.constants)
        end
        push!(cones_idxs, idxs)
        q += dim
        i += 1
    end
    if q > 0
        push!(cones, cone_from_moi(T, MOI.Nonnegatives(q)))
    end

    # other conic constraints
    opt.other_cones_start = i

    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !MOI.supports_constraint(opt, F, S)
            throw(MOI.UnsupportedConstraint{F,S}())
        end
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F,S}())
            if attr == MOI.ConstraintName() ||
               attr == MOI.ConstraintPrimalStart() ||
               attr == MOI.ConstraintDualStart()
                continue
            end
            throw(MOI.UnsupportedAttribute(attr))
        end
        if S == MOI.Zeros || S == MOI.Nonnegatives
            continue # already copied these constraints
        end
        @assert S <: SupportedCones{T}

        for ci in get_src_cons(F, S)
            idx_map[ci] = MOI.ConstraintIndex{F, S}(i)
            fi = get_con_fun(ci)
            si = get_con_set(ci)
            push!(other_cones, si)
            dim = MOI.output_dimension(fi)
            perm_idxs = q .+ permute_affine(si, 1:dim)

            if F == VV
                JGi = (idx_map[vj].value for vj in fi.variables)
                IGi = perm_idxs
                VGi = rescale_affine(si, fill(-one(T), dim))
            else
                JGi = (idx_map[vt.scalar_term.variable].value
                    for vt in fi.terms)
                IGi = permute_affine(si, [vt.output_index for vt in fi.terms])
                VGi = rescale_affine(si, [-vt.scalar_term.coefficient
                    for vt in fi.terms], IGi)
                IGi .+= q
                append!(Ih, perm_idxs)
                append!(Vh, rescale_affine(si, fi.constants))
            end

            append!(IG, IGi)
            append!(JG, JGi)
            append!(VG, VGi)
            push!(cones, cone_from_moi(T, si))
            push!(cones_idxs, q .+ (1:dim))
            q += dim
            i += 1
        end
    end

    # finalize model
    model_G = dropzeros!(sparse(IG, JG, VG, q, n))
    model_h = Vector(sparsevec(Ih, Vh, q))

    opt.model = Models.Model{T}(model_c, model_A, model_b, model_G, model_h,
        cones; obj_offset = obj_offset)

    opt.cones_idxs = cones_idxs
    opt.other_cones = other_cones
    return idx_map
end

function MOI.optimize!(opt::Optimizer{T}) where {T <: Real}
    # build and solve the model
    model = opt.model
    solver = opt.solver
    Solvers.load(solver, model)
    Solvers.solve(solver)

    opt.x = Solvers.get_x(solver)
    opt.y = Solvers.get_y(solver)
    opt.s = Solvers.get_s(solver)
    opt.z = Solvers.get_z(solver)

    # transform solution for MOI conventions
    opt.zeros_primal = copy(model.b)
    mul!(opt.zeros_primal, model.A, opt.x, -1, true)
    i = opt.other_cones_start
    for cone in opt.other_cones
        if needs_untransform(cone)
            idxs = opt.cones_idxs[i]
            @assert length(idxs) == MOI.dimension(cone)
            @views untransform_affine(cone, opt.s[idxs])
            @views untransform_affine(cone, opt.z[idxs])
        end
        i += 1
    end
    return
end

MOI.supports(::Optimizer, ::MOI.Silent) = true

function MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool)
    opt.solver.verbose = !value
    return
end

MOI.get(opt::Optimizer, ::MOI.Silent) = !opt.solver.verbose

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true

function MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, value::Union{Real, Nothing})
    opt.solver.time_limit = something(value, Inf)
    return
end

function MOI.get(opt::Optimizer, ::MOI.TimeLimitSec)
    if isfinite(opt.solver.time_limit)
        return opt.solver.time_limit
    end
    return
end

function MOI.get(opt::Optimizer, ::MOI.SolveTimeSec)
    if opt.solver.status in (:NotLoaded, :Loaded)
        error("solve has not been called")
    end
    return Solvers.get_solve_time(opt.solver)
end

MOI.get(opt::Optimizer, ::MOI.RawStatusString) = string(opt.solver.status)

MOI.get(opt::Optimizer, ::MOI.BarrierIterations) = opt.solver.num_iters

function MOI.set(opt::Optimizer, param::MOI.RawOptimizerAttribute, value)
    return setproperty!(opt.solver, Symbol(param.name), value)
end

function MOI.get(opt::Optimizer, param::MOI.RawOptimizerAttribute)
    return getproperty(opt.solver, Symbol(param.name))
end

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    status = opt.solver.status
    if status in (Solvers.NotLoaded, Solvers.Loaded)
        return MOI.OPTIMIZE_NOT_CALLED
    elseif status == Solvers.Optimal
        return MOI.OPTIMAL
    elseif status == Solvers.PrimalInfeasible || status == Solvers.PrimalInconsistent
        return MOI.INFEASIBLE
    elseif status == Solvers.DualInfeasible || status == Solvers.DualInconsistent
        return MOI.DUAL_INFEASIBLE
    elseif status == Solvers.SlowProgress
        return MOI.SLOW_PROGRESS
    elseif status == Solvers.IterationLimit
        return MOI.ITERATION_LIMIT
    elseif status == Solvers.TimeLimit
        return MOI.TIME_LIMIT
    else
        @warn("Hypatia status $(opt.solver.status) not handled")
        return MOI.OTHER_ERROR
    end
end

function MOI.get(opt::Optimizer, attr::MOI.PrimalStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    status = opt.solver.status
    if status == Solvers.Optimal
        return MOI.FEASIBLE_POINT
    elseif status == Solvers.PrimalInfeasible
        return MOI.INFEASIBLE_POINT
    elseif status == Solvers.DualInfeasible
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif status == Solvers.IllPosed
        return MOI.OTHER_RESULT_STATUS
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(opt::Optimizer, attr::MOI.DualStatus)
    if attr.result_index != 1
        return MOI.NO_SOLUTION
    end
    status = opt.solver.status
    if status == Solvers.Optimal
        return MOI.FEASIBLE_POINT
    elseif status == Solvers.PrimalInfeasible
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif status == Solvers.DualInfeasible
        return MOI.INFEASIBLE_POINT
    elseif status == Solvers.IllPosed
        return MOI.OTHER_RESULT_STATUS
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

_sense_val(sense::MOI.OptimizationSense) = (sense == MOI.MAX_SENSE ? -1 : 1)

function MOI.get(opt::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return _sense_val(opt.obj_sense) * Solvers.get_primal_obj(opt.solver)
end

function MOI.get(opt::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return _sense_val(opt.obj_sense) * Solvers.get_dual_obj(opt.solver)
end

MOI.get(opt::Optimizer, ::MOI.ResultCount) = 1

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vi::VI)
    MOI.check_result_index_bounds(opt, attr)
    return opt.x[vi.value]
end

function MOI.get(
    opt::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:Union{VV, VAF{T}}, MOI.Zeros},
    ) where T
    MOI.check_result_index_bounds(opt, attr)
    return opt.y[opt.zeros_idxs[ci.value]]
end

function MOI.get(
    opt::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:Union{VV, VAF{T}}, <:SupportedCones{T}},
    ) where T
    MOI.check_result_index_bounds(opt, attr)
    return opt.z[opt.cones_idxs[ci.value]]
end

function MOI.get(
    opt::Optimizer{T},
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{<:Union{VV, VAF{T}}, MOI.Zeros},
    ) where T
    MOI.check_result_index_bounds(opt, attr)
    return opt.zeros_primal[opt.zeros_idxs[ci.value]]
end

function MOI.get(
    opt::Optimizer{T},
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{<:Union{VV, VAF{T}}, <:SupportedCones{T}},
    ) where T
    MOI.check_result_index_bounds(opt, attr)
    return opt.s[opt.cones_idxs[ci.value]]
end
