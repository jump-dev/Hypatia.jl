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

    # data for transforming certificates
    obj_sense::MOI.OptimizationSense
    zeros_idxs::Vector{UnitRange{Int}}
    moi_cones::Vector{MOI.AbstractVectorSet}
    moi_cone_idxs::Vector{UnitRange{Int}}

    function Optimizer{T}(; options...) where {T <: Real}
        opt = new{T}()
        opt.solver = Solvers.Solver{T}(; options...)
        return opt
    end
end

Optimizer(; options...) = Optimizer{Float64}(; options...) # default to Float64

MOI.is_empty(opt::Optimizer) = (opt.solver.status == Solvers.NotLoaded)

MOI.empty!(opt::Optimizer) = (opt.solver.status = Solvers.NotLoaded)

MOI.get(::Optimizer, ::MOI.SolverName) = "Hypatia"

MOI.get(opt::Optimizer, ::MOI.RawSolver) = opt.solver

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
    if opt.solver.status == Solvers.NotLoaded
        error("solve has not been called")
    end
    return opt.solver.solve_time
end

MOI.get(opt::Optimizer, ::MOI.RawStatusString) = string(opt.solver.status)

MOI.get(opt::Optimizer, ::MOI.BarrierIterations) = opt.solver.num_iters

function MOI.set(opt::Optimizer, param::MOI.RawOptimizerAttribute, value)
    return setproperty!(opt.solver, Symbol(param.name), value)
end

function MOI.get(opt::Optimizer, param::MOI.RawOptimizerAttribute)
    return getproperty(opt.solver, Symbol(param.name))
end

MOI.supports(
    ::Optimizer{T},
    ::Union{MOI.ObjectiveSense, MOI.ObjectiveFunction{<:Union{VI, SAF{T}}}},
    ) where {T <: Real} = true

MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{<:Union{VV, VAF{T}}},
    ::Type{<:Union{MOI.Zeros, SupportedCone{T}}},
    ) where {T <: Real} = true

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
    model_c = zeros(T, n)
    obj_offset = 0.0
    for attr in MOI.get(src, MOI.ListOfModelAttributesSet())
        if attr == MOI.Name()
            continue
        elseif attr == MOI.ObjectiveSense()
            opt.obj_sense = MOI.get(src, MOI.ObjectiveSense())
        elseif attr isa MOI.ObjectiveFunction
            F = MOI.get(src, MOI.ObjectiveFunctionType())
            if !(F <: Union{VI, SAF{T}})
                error("objective function type $F not supported")
            end
            obj = convert(SAF{T}, MOI.get(src, MOI.ObjectiveFunction{F}()))
            for t in obj.terms
                model_c[idx_map[t.variable].value] += t.coefficient
            end
            obj_offset = obj.constant
        else
            throw(MOI.UnsupportedAttribute(attr))
        end
    end
    if opt.obj_sense == MOI.MAX_SENSE
        model_c .*= -1
        obj_offset *= -1
    end

    # constraints
    get_src_cons(F, S) = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    get_con_fun(con_idx) = MOI.get(src, MOI.ConstraintFunction(), con_idx)
    get_con_set(con_idx) = MOI.get(src, MOI.ConstraintSet(), con_idx)

    # equality constraints
    i = 1 # MOI constraint index
    p = 0 # rows of A (equality constraint matrix)
    (IA, JA, VA) = (Int[], Int[], T[])
    model_b = T[]
    zeros_idxs = Vector{UnitRange{Int}}()

    for F in (VV, VAF{T}), ci in get_src_cons(F, MOI.Zeros)
        idx_map[ci] = MOI.ConstraintIndex{F, MOI.Zeros}(i)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        _con_IJV(IA, JA, VA, model_b, fi, p, dim, idx_map)
        push!(zeros_idxs, p .+ (1:dim))
        p += dim
        i += 1
    end

    model_A = dropzeros!(sparse(IA, JA, VA, p, n))
    opt.zeros_idxs = zeros_idxs

    # conic constraints
    moi_cones = MOI.AbstractVectorSet[]
    i = 1 # MOI constraint index
    q = 0 # rows of G (cone constraint matrix)
    (IG, JG, VG) = (Int[], Int[], T[])
    model_h = T[]
    moi_cone_idxs = Vector{UnitRange{Int}}()
    cones = Cones.Cone{T}[]

    # build up one nonnegative cone
    for F in (VV, VAF{T}), ci in get_src_cons(F, MOI.Nonnegatives)
        idx_map[ci] = MOI.ConstraintIndex{F, MOI.Nonnegatives}(i)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        _con_IJV(IG, JG, VG, model_h, fi, q, dim, idx_map)
        push!(moi_cones, get_con_set(ci))
        push!(moi_cone_idxs, q .+ (1:dim))
        q += dim
        i += 1
    end
    if q > 0
        push!(cones, cone_from_moi(T, MOI.Nonnegatives(q)))
    end

    # other conic constraints
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
        @assert S <: SupportedCone{T}

        for ci in get_src_cons(F, S)
            idx_map[ci] = MOI.ConstraintIndex{F, S}(i)
            fi = get_con_fun(ci)
            si = get_con_set(ci)
            dim = MOI.output_dimension(fi)

            perm_idxs = permute_affine(si, 1:dim)
            if F == VV
                JGi = (idx_map[vj].value for vj in fi.variables)
                IGi = q .+ perm_idxs
                VGi = rescale_affine(si, fill(-one(T), dim))
                append!(model_h, zero(T) for _ in 1:dim)
            else
                JGi = (idx_map[vt.scalar_term.variable].value
                    for vt in fi.terms)
                IGi = permute_affine(si, [vt.output_index for vt in fi.terms])
                VGi = rescale_affine(si, [-vt.scalar_term.coefficient
                    for vt in fi.terms], IGi)
                IGi .+= q
                hi = zeros(T, dim)
                hi[perm_idxs] = rescale_affine(si, fi.constants)
                append!(model_h, hi)
            end
            append!(IG, IGi)
            append!(JG, JGi)
            append!(VG, VGi)

            push!(cones, cone_from_moi(T, si))
            push!(moi_cones, si)
            push!(moi_cone_idxs, q .+ (1:dim))

            q += dim
            i += 1
        end
    end

    model_G = dropzeros!(sparse(IG, JG, VG, q, n))
    opt.moi_cone_idxs = moi_cone_idxs
    opt.moi_cones = moi_cones

    # finalize model and load into solver
    model = Models.Model{T}(model_c, model_A, model_b, model_G, model_h,
        cones; obj_offset = obj_offset)
    Solvers.load(opt.solver, model)

    return idx_map
end

MOI.optimize!(opt::Optimizer) = Solvers.solve(opt.solver)

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
    return _sense_val(opt.obj_sense) * opt.solver.primal_obj
end

function MOI.get(opt::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(opt, attr)
    return _sense_val(opt.obj_sense) * opt.solver.dual_obj
end

MOI.get(opt::Optimizer, ::MOI.ResultCount) = 1

function MOI.get(opt::Optimizer, attr::MOI.VariablePrimal, vi::VI)
    MOI.check_result_index_bounds(opt, attr)
    return opt.solver.result.x[vi.value]
end

function MOI.get(
    opt::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:Union{VV, VAF{T}}, MOI.Zeros},
    ) where {T}
    MOI.check_result_index_bounds(opt, attr)
    return opt.solver.result.y[opt.zeros_idxs[ci.value]]
end

function MOI.get(
    opt::Optimizer{T},
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex{<:Union{VV, VAF{T}}, <:SupportedCone{T}},
    ) where {T}
    MOI.check_result_index_bounds(opt, attr)
    i = ci.value
    z_i = opt.solver.result.z[opt.moi_cone_idxs[i]]
    return _transform_sz(z_i, opt.moi_cones[i])
end

function MOI.get(
    opt::Optimizer{T},
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{<:Union{VV, VAF{T}}, <:SupportedCone{T}},
    ) where {T}
    MOI.check_result_index_bounds(opt, attr)
    i = ci.value
    s_i = opt.solver.result.s[opt.moi_cone_idxs[i]]
    return _transform_sz(s_i, opt.moi_cones[i])
end

function _transform_sz(sz::Vector{T}, cone::SupportedCone{T}) where {T}
    if needs_untransform(cone)
        @assert length(sz) == MOI.dimension(cone)
        untransform_affine(cone, sz)
    end
    return sz
end



# TODO don't use sparsevec for b and h - append on a vector each time
function _con_IJV(
    IM::Vector{Int},
    JM::Vector{Int},
    VM::Vector{T},
    vect::Vector{T},
    func::VV,
    start::Int,
    dim::Int,
    idx_map::MOI.IndexMap,
    ) where {T <: Real}
    append!(IM, start .+ (1:dim))
    append!(JM, idx_map[vi].value for vi in func.variables)
    append!(VM, -one(T) for _ in 1:dim)
    append!(vect, zero(T) for _ in 1:dim)
    return
end

function _con_IJV(
    IM::Vector{Int},
    JM::Vector{Int},
    VM::Vector{T},
    vect::Vector{T},
    func::VAF{T},
    start::Int,
    dim::Int,
    idx_map::MOI.IndexMap,
    ) where {T <: Real}
    append!(IM, start + vt.output_index for vt in func.terms)
    append!(JM, idx_map[vt.scalar_term.variable].value for vt in func.terms)
    append!(VM, -vt.scalar_term.coefficient for vt in func.terms)
    append!(vect, func.constants)
    return
end