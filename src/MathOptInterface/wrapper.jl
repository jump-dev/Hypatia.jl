#=
Copyright (c) 2018-2022 Chris Coey, Lea Kapelevich, and contributors

This Julia package Hypatia.jl is released under the MIT license; see LICENSE
file in the root directory or at https://github.com/chriscoey/Hypatia.jl
=#

#=
MathOptInterface wrapper of Hypatia solver
=#

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

function MOI.supports(
    ::Optimizer{T},
    ::Union{MOI.ObjectiveSense, MOI.ObjectiveFunction{<:Union{VI, SAF{T}}}},
) where {T <: Real}
    return true
end

function MOI.supports_constraint(
    ::Optimizer{T},
    ::Type{<:Union{VV, VAF{T}}},
    ::Type{<:Union{MOI.Zeros, SupportedCone{T}}},
) where {T <: Real}
    return true
end

function MOI.copy_to(opt::Optimizer{T}, src::MOI.ModelLike) where {T <: Real}
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
    obj_offset = zero(T)
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
    (IA, JA, VA) = (Int[], Int[], T[])
    model_b = T[]
    opt.zeros_idxs = zeros_idxs = Vector{UnitRange{Int}}()
    for F in (VV, VAF{T}), ci in get_src_cons(F, MOI.Zeros)
        fi = get_con_fun(ci)
        si = get_con_set(ci)
        _con_IJV(IA, JA, VA, model_b, zeros_idxs, fi, si, idx_map)
        idx_map[ci] = MOI.ConstraintIndex{F, MOI.Zeros}(length(zeros_idxs))
    end
    model_A = dropzeros!(sparse(IA, JA, VA, length(model_b), n))

    # conic constraints
    (IG, JG, VG) = (Int[], Int[], T[])
    model_h = T[]
    moi_cones = MOI.AbstractVectorSet[]
    moi_cone_idxs = Vector{UnitRange{Int}}()
    cones = Cones.Cone{T}[]

    # build up one nonnegative cone
    for F in (VV, VAF{T}), ci in get_src_cons(F, MOI.Nonnegatives)
        fi = get_con_fun(ci)
        si = get_con_set(ci)
        _con_IJV(IG, JG, VG, model_h, moi_cone_idxs, fi, si, idx_map)
        push!(moi_cones, si)
        idx_map[ci] = MOI.ConstraintIndex{F, MOI.Nonnegatives}(length(moi_cones))
    end
    if !isempty(moi_cones)
        push!(cones, cone_from_moi(T, MOI.Nonnegatives(length(model_h))))
    end

    # other conic constraints
    for (F, S) in MOI.get(src, MOI.ListOfConstraintTypesPresent())
        if !MOI.supports_constraint(opt, F, S)
            throw(MOI.UnsupportedConstraint{F, S}())
        end
        for attr in MOI.get(src, MOI.ListOfConstraintAttributesSet{F, S}())
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

        for ci in get_src_cons(F, S)
            fi = get_con_fun(ci)
            si = get_con_set(ci)
            _con_IJV(IG, JG, VG, model_h, moi_cone_idxs, fi, si, idx_map)
            push!(cones, cone_from_moi(T, si))
            push!(moi_cones, si)
            idx_map[ci] = MOI.ConstraintIndex{F, S}(length(moi_cones))
        end
    end

    model_G = dropzeros!(sparse(IG, JG, VG, length(model_h), n))
    opt.moi_cone_idxs = moi_cone_idxs
    opt.moi_cones = moi_cones

    # finalize model and load into solver
    model = Models.Model{T}(
        model_c,
        model_A,
        model_b,
        model_G,
        model_h,
        cones;
        obj_offset = obj_offset,
    )
    Solvers.load(opt.solver, model)

    return idx_map
end

MOI.optimize!(opt::Optimizer) = Solvers.solve(opt.solver)

function MOI.modify(
    opt::Optimizer{T},
    ::MOI.ObjectiveFunction{SAF{T}},
    chg::MOI.ScalarConstantChange{T},
) where {T}
    obj_offset = chg.new_constant
    if opt.obj_sense == MOI.MAX_SENSE
        obj_offset = -obj_offset
    end
    Solvers.modify_obj_offset(opt.solver, obj_offset)
    return
end

function MOI.modify(
    opt::Optimizer{T},
    ::MOI.ObjectiveFunction{SAF{T}},
    chg::MOI.ScalarCoefficientChange{T},
) where {T}
    new_c = chg.new_coefficient
    if opt.obj_sense == MOI.MAX_SENSE
        new_c = -new_c
    end
    Solvers.modify_c(opt.solver, [chg.variable.value], [new_c])
    return
end

function MOI.modify(
    opt::Optimizer{T},
    ci::MOI.ConstraintIndex{VAF{T}, MOI.Zeros},
    chg::MOI.VectorConstantChange{T},
) where {T}
    idxs = opt.zeros_idxs[ci.value]
    Solvers.modify_b(opt.solver, idxs, chg.new_constant)
    return
end

function MOI.modify(
    opt::Optimizer{T},
    ci::MOI.ConstraintIndex{VAF{T}, <:SupportedCone{T}},
    chg::MOI.VectorConstantChange{T},
) where {T}
    i = ci.value
    idxs = opt.moi_cone_idxs[i]
    set = opt.moi_cones[i]
    new_h = chg.new_constant
    if needs_permute(set)
        @assert !needs_rescale(set)
        new_h = permute_affine(set, new_h)
    end
    if needs_rescale(set)
        rescale_affine(set, new_h)
    end
    Solvers.modify_h(opt.solver, idxs, new_h)
    return
end

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    status = opt.solver.status
    if status in (Solvers.NotLoaded, Solvers.Loaded)
        return MOI.OPTIMIZE_NOT_CALLED
    elseif status == Solvers.Optimal
        return MOI.OPTIMAL
    elseif status == Solvers.NearOptimal
        return MOI.ALMOST_OPTIMAL
    elseif status == Solvers.PrimalInfeasible || status == Solvers.PrimalInconsistent
        return MOI.INFEASIBLE
    elseif status == Solvers.NearPrimalInfeasible
        return MOI.ALMOST_INFEASIBLE
    elseif status == Solvers.DualInfeasible || status == Solvers.DualInconsistent
        return MOI.DUAL_INFEASIBLE
    elseif status == Solvers.NearDualInfeasible
        return MOI.ALMOST_DUAL_INFEASIBLE
    elseif status == Solvers.SlowProgress
        return MOI.SLOW_PROGRESS
    elseif status == Solvers.IterationLimit
        return MOI.ITERATION_LIMIT
    elseif status == Solvers.TimeLimit
        return MOI.TIME_LIMIT
    elseif status == Solvers.NumericalFailure
        return MOI.NUMERICAL_ERROR
    elseif status in (Solvers.IllPosed, Solvers.NearIllPosed)
        return MOI.OTHER_LIMIT
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
    elseif status == Solvers.NearOptimal
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == Solvers.PrimalInfeasible
        return MOI.INFEASIBLE_POINT
    elseif status == Solvers.DualInfeasible
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif status == Solvers.NearDualInfeasible
        return MOI.NEARLY_INFEASIBILITY_CERTIFICATE
    elseif status in (Solvers.IllPosed, Solvers.NearIllPosed)
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
    elseif status == Solvers.NearOptimal
        return MOI.NEARLY_FEASIBLE_POINT
    elseif status == Solvers.PrimalInfeasible
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif status == Solvers.NearPrimalInfeasible
        return MOI.NEARLY_INFEASIBILITY_CERTIFICATE
    elseif status == Solvers.DualInfeasible
        return MOI.INFEASIBLE_POINT
    elseif status in (Solvers.IllPosed, Solvers.NearIllPosed)
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
    return untransform_affine(opt.moi_cones[i], z_i)
end

function MOI.get(
    opt::Optimizer{T},
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex{<:Union{VV, VAF{T}}, <:SupportedCone{T}},
) where {T}
    MOI.check_result_index_bounds(opt, attr)
    i = ci.value
    s_i = opt.solver.result.s[opt.moi_cone_idxs[i]]
    return untransform_affine(opt.moi_cones[i], s_i)
end

function _con_IJV(
    IM::Vector{Int},
    JM::Vector{Int},
    VM::Vector{T},
    vect::Vector{T},
    idxs_vect::Vector{UnitRange{Int}},
    func::VV,
    set::MOI.AbstractVectorSet,
    idx_map::MOI.IndexMap,
) where {T <: Real}
    dim = MOI.output_dimension(func)
    start = length(vect)
    idxs = start .+ (1:dim)
    push!(idxs_vect, idxs)
    append!(vect, zero(T) for _ in 1:dim)
    if needs_permute(set)
        perm_idxs = permute_affine(set, 1:dim)
        perm_idxs .+= start
        append!(IM, perm_idxs)
    else
        append!(IM, idxs)
    end
    append!(JM, idx_map[vi].value for vi in func.variables)
    append!(VM, -one(T) for _ in 1:dim)
    if needs_rescale(set)
        @views rescale_affine(set, VM[(end - dim + 1):end])
    end
    return
end

function _con_IJV(
    IM::Vector{Int},
    JM::Vector{Int},
    VM::Vector{T},
    vect::Vector{T},
    idxs_vect::Vector{UnitRange{Int}},
    func::VAF{T},
    set::MOI.AbstractVectorSet,
    idx_map::MOI.IndexMap,
) where {T <: Real}
    dim = MOI.output_dimension(func)
    start = length(vect)
    idxs = start .+ (1:dim)
    push!(idxs_vect, idxs)
    if needs_permute(set)
        @assert !needs_rescale(set)
        append!(vect, permute_affine(set, func.constants))
        perm_idxs = permute_affine(set, func)
        perm_idxs .+= start
        append!(IM, perm_idxs)
    else
        append!(vect, func.constants)
        append!(IM, start + vt.output_index for vt in func.terms)
    end
    append!(JM, idx_map[vt.scalar_term.variable].value for vt in func.terms)
    append!(VM, -vt.scalar_term.coefficient for vt in func.terms)
    if needs_rescale(set)
        @views vm = VM[(end - length(func.terms) + 1):end]
        rescale_affine(set, func, vm)
        @views rescale_affine(set, vect[idxs])
    end
    return
end
