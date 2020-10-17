#=
MathOptInterface wrapper of Hypatia solver

TODO generalize all code for T <: Real
=#

mutable struct Optimizer{T <: Real} <: MOI.AbstractOptimizer
    use_dense_model::Bool # make the model use dense A and G data instead of sparse

    solver::Solvers.Solver{T} # Hypatia solver object
    model::Models.Model{T} # Hypatia model object

    # result data
    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}

    # data for transforming certificates
    obj_sense::MOI.OptimizationSense
    num_eq_constrs::Int
    constr_offset_eq::Vector{Int}
    constr_prim_eq::Vector{T}
    constr_offset_cone::Vector{Int}
    constr_prim_cone::Vector{T}
    nonpos_idxs::UnitRange{Int}
    interval_idxs::UnitRange{Int}
    interval_scales::Vector{T}
    moi_other_cones_start::Int
    moi_other_cones::Vector{MOI.AbstractVectorSet}

    function Optimizer{T}(;
        use_dense_model::Bool = true,
        solver_options...
        ) where {T <: Real}
        opt = new{T}()
        opt.use_dense_model = use_dense_model
        opt.solver = Solvers.Solver{T}(; solver_options...) # TODO allow passing in a solver?
        return opt
    end
end

Optimizer(; options...) = Optimizer{Float64}(; options...) # default to Float64

MOI.get(::Optimizer, ::MOI.SolverName) = "Hypatia"
MOI.get(opt::Optimizer, ::MOI.RawSolver) = opt.solver

MOI.is_empty(opt::Optimizer) = (opt.solver.status == Solvers.NotLoaded)

function MOI.empty!(opt::Optimizer)
    opt.solver.status = Solvers.NotLoaded
    return
end

MOI.supports(::Optimizer{T}, ::Union{
    MOI.ObjectiveSense,
    MOI.ObjectiveFunction{MOI.SingleVariable},
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}},
    ) where {T <: Real} = true

MOI.supports_constraint(::Optimizer{T},
    ::Type{<:Union{MOI.SingleVariable, MOI.ScalarAffineFunction{T}}},
    ::Type{<:Union{MOI.EqualTo{T}, MOI.GreaterThan{T}, MOI.LessThan{T}, MOI.Interval{T}}},
    ) where {T <: Real} = true
MOI.supports_constraint(::Optimizer{T},
    ::Type{<:Union{MOI.VectorOfVariables, MOI.VectorAffineFunction{T}}},
    ::Type{<:SupportedCones{T}},
    ) where {T <: Real} = true

# build representation as min c'x s.t. A*x = b, h - G*x in K
function MOI.copy_to(
    opt::Optimizer{T},
    src::MOI.ModelLike;
    copy_names::Bool = false,
    warn_attributes::Bool = true,
    ) where {T <: Real}
    @assert !copy_names
    idx_map = MOI.Utilities.IndexMap()

    # variables
    n = MOI.get(src, MOI.NumberOfVariables()) # columns of A
    j = 0
    for vj in MOI.get(src, MOI.ListOfVariableIndices()) # MOI.VariableIndex
        j += 1
        idx_map[vj] = MOI.VariableIndex(j)
    end
    @assert j == n

    # objective function
    F = MOI.get(src, MOI.ObjectiveFunctionType())
    if F == MOI.SingleVariable
        obj = MOI.ScalarAffineFunction{T}(MOI.get(src, MOI.ObjectiveFunction{F}()))
    elseif F == MOI.ScalarAffineFunction{T}
        obj = MOI.get(src, MOI.ObjectiveFunction{F}())
    else
        error("function type $F not supported")
    end
    (Jc, Vc) = (Int[], T[])
    for t in obj.terms
        push!(Jc, idx_map[t.variable_index].value)
        push!(Vc, t.coefficient)
    end
    obj_offset = obj.constant
    if MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        Vc .*= -1
        obj_offset *= -1
    end
    opt.obj_sense = MOI.get(src, MOI.ObjectiveSense())
    model_c = Vector(sparsevec(Jc, Vc, n))

    # constraints
    get_src_cons(F, S) = MOI.get(src, MOI.ListOfConstraintIndices{F, S}())
    get_con_fun(con_idx) = MOI.get(src, MOI.ConstraintFunction(), con_idx)
    get_con_set(con_idx) = MOI.get(src, MOI.ConstraintSet(), con_idx)
    i = 0 # MOI constraint index

    # equality constraints
    p = 0 # rows of A (equality constraint matrix)
    (IA, JA, VA) = (Int[], Int[], T[])
    (Ib, Vb) = (Int[], T[])
    (Icpe, Vcpe) = (Int[], T[]) # constraint set constants for opt.constr_prim_eq
    constr_offset_eq = Vector{Int}()

    for ci in get_src_cons(MOI.SingleVariable, MOI.EqualTo{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{T}}(i)
        push!(constr_offset_eq, p)
        p += 1
        push!(IA, p)
        push!(JA, idx_map[get_con_fun(ci).variable].value)
        push!(VA, -1)
        push!(Ib, p)
        push!(Vb, -get_con_set(ci).value)
        push!(Icpe, p)
        push!(Vcpe, get_con_set(ci).value)
    end

    for ci in get_src_cons(MOI.ScalarAffineFunction{T}, MOI.EqualTo{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.EqualTo{T}}(i)
        push!(constr_offset_eq, p)
        p += 1
        fi = get_con_fun(ci)
        for vt in fi.terms
            push!(IA, p)
            push!(JA, idx_map[vt.variable_index].value)
            push!(VA, -vt.coefficient)
        end
        push!(Ib, p)
        push!(Vb, fi.constant - get_con_set(ci).value)
        push!(Icpe, p)
        push!(Vcpe, get_con_set(ci).value)
    end

    for ci in get_src_cons(MOI.VectorOfVariables, MOI.Zeros)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Zeros}(i)
        push!(constr_offset_eq, p)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        append!(IA, (p + 1):(p + dim))
        append!(JA, idx_map[vi].value for vi in fi.variables)
        append!(VA, -ones(T, dim))
        p += dim
    end

    for ci in get_src_cons(MOI.VectorAffineFunction{T}, MOI.Zeros)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{T}, MOI.Zeros}(i)
        push!(constr_offset_eq, p)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IA, p + vt.output_index)
            push!(JA, idx_map[vt.scalar_term.variable_index].value)
            push!(VA, -vt.scalar_term.coefficient)
        end
        append!(Ib, (p + 1):(p + dim))
        append!(Vb, fi.constants)
        p += dim
    end

    push!(constr_offset_eq, p)
    model_A = dropzeros!(sparse(IA, JA, VA, p, n))
    model_b = Vector(sparsevec(Ib, Vb, p))
    opt.num_eq_constrs = i
    opt.constr_prim_eq = Vector(sparsevec(Icpe, Vcpe, p))
    opt.constr_offset_eq = constr_offset_eq

    # conic constraints
    q = 0 # rows of G (cone constraint matrix)
    (IG, JG, VG) = (Int[], Int[], T[])
    (Ih, Vh) = (Int[], T[])
    (Icpc, Vcpc) = (Int[], T[]) # constraint set constants for opt.constr_prim_eq
    constr_offset_cone = Vector{Int}()
    cones = Cones.Cone{T}[]

    # build up one nonnegative cone from LP constraints
    nonneg_start = q

    # nonnegative-like constraints
    for ci in get_src_cons(MOI.SingleVariable, MOI.GreaterThan{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{T}}(i)
        push!(constr_offset_cone, q)
        q += 1
        push!(IG, q)
        push!(JG, idx_map[get_con_fun(ci).variable].value)
        push!(VG, -1)
        push!(Ih, q)
        push!(Vh, -get_con_set(ci).lower)
        push!(Vcpc, get_con_set(ci).lower)
        push!(Icpc, q)
    end

    for ci in get_src_cons(MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.GreaterThan{T}}(i)
        push!(constr_offset_cone, q)
        q += 1
        fi = get_con_fun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idx_map[vt.variable_index].value)
            push!(VG, -vt.coefficient)
        end
        push!(Ih, q)
        push!(Vh, fi.constant - get_con_set(ci).lower)
        push!(Vcpc, get_con_set(ci).lower)
        push!(Icpc, q)
    end

    for ci in get_src_cons(MOI.VectorOfVariables, MOI.Nonnegatives)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonnegatives}(i)
        push!(constr_offset_cone, q)
        for vj in get_con_fun(ci).variables
            q += 1
            push!(IG, q)
            push!(JG, idx_map[vj].value)
            push!(VG, -1)
        end
    end

    for ci in get_src_cons(MOI.VectorAffineFunction{T}, MOI.Nonnegatives)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{T}, MOI.Nonnegatives}(i)
        push!(constr_offset_cone, q)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IG, q + vt.output_index)
            push!(JG, idx_map[vt.scalar_term.variable_index].value)
            push!(VG, -vt.scalar_term.coefficient)
        end
        append!(Ih, (q + 1):(q + dim))
        append!(Vh, fi.constants)
        q += dim
    end

    # nonpositive-like constraints
    nonpos_start = q

    for ci in get_src_cons(MOI.SingleVariable, MOI.LessThan{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{T}}(i)
        push!(constr_offset_cone, q)
        q += 1
        push!(IG, q)
        push!(JG, idx_map[get_con_fun(ci).variable].value)
        push!(VG, 1)
        push!(Ih, q)
        push!(Vh, get_con_set(ci).upper)
        push!(Vcpc, get_con_set(ci).upper)
        push!(Icpc, q)
    end

    for ci in get_src_cons(MOI.ScalarAffineFunction{T}, MOI.LessThan{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.LessThan{T}}(i)
        push!(constr_offset_cone, q)
        q += 1
        fi = get_con_fun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idx_map[vt.variable_index].value)
            push!(VG, vt.coefficient)
        end
        push!(Ih, q)
        push!(Vh, -fi.constant + get_con_set(ci).upper)
        push!(Vcpc, get_con_set(ci).upper)
        push!(Icpc, q)
    end

    for ci in get_src_cons(MOI.VectorOfVariables, MOI.Nonpositives)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorOfVariables, MOI.Nonpositives}(i)
        push!(constr_offset_cone, q)
        for vj in get_con_fun(ci).variables
            q += 1
            push!(IG, q)
            push!(JG, idx_map[vj].value)
            push!(VG, 1)
        end
    end

    for ci in get_src_cons(MOI.VectorAffineFunction{T}, MOI.Nonpositives)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{T}, MOI.Nonpositives}(i)
        push!(constr_offset_cone, q)
        fi = get_con_fun(ci)
        dim = MOI.output_dimension(fi)
        for vt in fi.terms
            push!(IG, q + vt.output_index)
            push!(JG, idx_map[vt.scalar_term.variable_index].value)
            push!(VG, vt.scalar_term.coefficient)
        end
        append!(Ih, (q + 1):(q + dim))
        append!(Vh, -fi.constants)
        q += dim
    end

    # single nonnegative cone
    opt.nonpos_idxs = (nonpos_start + 1):q
    if q > nonneg_start
        push!(cones, Cones.Nonnegative{T}(q - nonneg_start))
    end

    # build up one L_infinity norm cone from two-sided interval constraints
    interval_start = q
    SV_interval_cons = get_src_cons(MOI.SingleVariable, MOI.Interval{T})
    SAF_interval_cons = get_src_cons(MOI.ScalarAffineFunction{T}, MOI.Interval{T})
    num_intervals = length(SV_interval_cons) + length(SAF_interval_cons)
    interval_scales = zeros(T, num_intervals)

    if num_intervals > 0
        i += 1
        push!(constr_offset_cone, q)
        q += 1
        push!(Ih, q)
        push!(Vh, 1)
    end

    interval_count = 0

    for ci in SV_interval_cons
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{T}}(i)
        push!(constr_offset_cone, q)
        q += 1

        upper = get_con_set(ci).upper
        lower = get_con_set(ci).lower
        @assert isfinite(upper) && isfinite(lower)
        @assert upper > lower
        mid = (upper + lower) / T(2)
        scal = 2 / (upper - lower)

        push!(IG, q)
        push!(JG, idx_map[get_con_fun(ci).variable].value)
        push!(VG, -scal)
        push!(Ih, q)
        push!(Vh, -mid * scal)
        push!(Vcpc, mid)
        push!(Icpc, q)
        interval_count += 1
        interval_scales[interval_count] = scal
    end

    for ci in SAF_interval_cons
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.Interval{T}}(i)
        push!(constr_offset_cone, q)
        q += 1

        upper = get_con_set(ci).upper
        lower = get_con_set(ci).lower
        @assert isfinite(upper) && isfinite(lower)
        @assert upper > lower
        mid = (upper + lower) / T(2)
        scal = 2 / (upper - lower)

        fi = get_con_fun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idx_map[vt.variable_index].value)
            push!(VG, -vt.coefficient * scal)
        end
        push!(Ih, q)
        push!(Vh, (fi.constant - mid) * scal)
        push!(Vcpc, mid)
        push!(Icpc, q)
        interval_count += 1
        interval_scales[interval_count] = scal
    end

    opt.interval_idxs = (interval_start + 2):q
    opt.interval_scales = interval_scales
    if q > interval_start
        # exists at least one interval-type constraint
        push!(cones, Cones.EpiNormInf{T, T}(q - interval_start))
    end

    # non-LP conic constraints
    moi_other_cones_start = i + 1
    moi_other_cones = MOI.AbstractVectorSet[]

    for (F, S) in MOI.get(src, MOI.ListOfConstraints())
        if S <: Union{MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives}
            continue # already copied these constraints
        end
        @assert S <: SupportedCones{T}
        for ci in get_src_cons(F, S)
            i += 1
            idx_map[ci] = MOI.ConstraintIndex{F, S}(i)
            push!(constr_offset_cone, q)
            fi = get_con_fun(ci)
            si = get_con_set(ci)
            push!(moi_other_cones, si)
            dim = MOI.output_dimension(fi)
            if F == MOI.VectorOfVariables
                JGi = (idx_map[vj].value for vj in fi.variables)
                IGi = permute_affine(si, 1:dim)
                VGi = rescale_affine(si, fill(-one(T), dim))
            else
                JGi = (idx_map[vt.scalar_term.variable_index].value for vt in fi.terms)
                IGi = permute_affine(si, [vt.output_index for vt in fi.terms])
                VGi = rescale_affine(si, [-vt.scalar_term.coefficient for vt in fi.terms], IGi)
                Ihi = permute_affine(si, 1:dim)
                Vhi = rescale_affine(si, fi.constants)
                Ihi = q .+ Ihi
                append!(Ih, Ihi)
                append!(Vh, Vhi)
            end
            IGi = q .+ IGi
            append!(IG, IGi)
            append!(JG, JGi)
            append!(VG, VGi)
            push!(cones, cone_from_moi(T, si))
            q += dim
        end
    end

    push!(constr_offset_cone, q)

    # finalize model
    model_G = dropzeros!(sparse(IG, JG, VG, q, n))
    model_h = Vector(sparsevec(Ih, Vh, q))

    opt.model = Models.Model{T}(model_c, model_A, model_b, model_G, model_h, cones; obj_offset = obj_offset)
    if opt.use_dense_model # convert A and G to dense
        Models.densify!(opt.model)
    end

    opt.constr_offset_cone = constr_offset_cone
    opt.constr_prim_cone = Vector(sparsevec(Icpc, Vcpc, q))
    opt.moi_other_cones_start = moi_other_cones_start
    opt.moi_other_cones = moi_other_cones

    return idx_map
end

function MOI.optimize!(opt::Optimizer{T}) where {T <: Real}
    # build and solve the model
    model = opt.model
    solver = opt.solver

    Solvers.load(solver, model)
    Solvers.solve(solver)

    status = Solvers.get_status(solver)
    primal_obj = Solvers.get_primal_obj(solver)
    dual_obj = Solvers.get_dual_obj(solver)
    opt.x = Solvers.get_x(solver)
    opt.y = Solvers.get_y(solver)
    opt.s = Solvers.get_s(solver)
    opt.z = Solvers.get_z(solver)

    # transform solution for MOI conventions
    opt.constr_prim_eq += model.b - model.A * opt.x
    opt.s[opt.nonpos_idxs] .*= -1
    opt.z[opt.nonpos_idxs] .*= -1
    opt.s[opt.interval_idxs] ./= opt.interval_scales
    i = opt.moi_other_cones_start - opt.num_eq_constrs
    for cone in opt.moi_other_cones
        if needs_untransform(cone)
            os = opt.constr_offset_cone
            idxs = (os[i] + 1):os[i + 1]
            @views untransform_affine(cone, opt.s[idxs])
            @views untransform_affine(cone, opt.z[idxs])
        end
        i += 1
    end
    opt.constr_prim_cone .+= opt.s
    opt.z[opt.interval_idxs] .*= opt.interval_scales

    return
end

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool) = (opt.solver.verbose = value)
MOI.get(opt::Optimizer, ::MOI.Silent) = opt.solver.verbose

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, value::Real) = (opt.solver.time_limit = value)
MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, ::Nothing) = (opt.solver.time_limit = Inf)
MOI.get(opt::Optimizer, ::MOI.TimeLimitSec) = (isfinite(opt.solver.time_limit) ? opt.solver.time_limit : nothing)

function MOI.get(opt::Optimizer, ::MOI.SolveTime)
    if opt.solver.status in (:NotLoaded, :Loaded)
        error("solve has not been called")
    end
    return Solvers.get_solve_time(opt.solver)
end

MOI.get(opt::Optimizer, ::MOI.RawStatusString) = string(opt.solver.status)

MOI.get(opt::Optimizer, ::MOI.BarrierIterations) = opt.solver.num_iters

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    status = opt.solver.status
    if status in (Solvers.NotLoaded, Solvers.Loaded)
        return MOI.OPTIMIZE_NOT_CALLED
    elseif status == Solvers.Optimal
        return MOI.OPTIMAL
    elseif status == Solvers.PrimalInfeasible
        return MOI.INFEASIBLE
    elseif status == Solvers.DualInfeasible
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

function MOI.get(opt::Optimizer, ::MOI.PrimalStatus)
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

function MOI.get(opt::Optimizer, ::MOI.DualStatus)
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

function MOI.get(opt::Optimizer, ::MOI.ObjectiveValue)
    raw_obj_val = Solvers.get_primal_obj(opt.solver)
    return (opt.obj_sense == MOI.MAX_SENSE) ? -raw_obj_val : raw_obj_val
end

function MOI.get(opt::Optimizer, ::Union{MOI.DualObjectiveValue, MOI.ObjectiveBound})
    raw_dual_obj_val = Solvers.get_dual_obj(opt.solver)
    return (opt.obj_sense == MOI.MAX_SENSE) ? -raw_dual_obj_val : raw_dual_obj_val
end

MOI.get(opt::Optimizer, ::MOI.ResultCount) = 1

MOI.get(opt::Optimizer, ::MOI.VariablePrimal, vi::MOI.VariableIndex) = opt.x[vi.value]
MOI.get(opt::Optimizer, a::MOI.VariablePrimal, vi::Vector{MOI.VariableIndex}) = MOI.get.(opt, a, vi)

function MOI.get(opt::Optimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractScalarSet}
    # scalar set
    i = ci.value
    if i <= opt.num_eq_constrs
        # constraint is an equality
        return opt.y[opt.constr_offset_eq[i] + 1]
    else
        # constraint is conic
        i -= opt.num_eq_constrs
        return opt.z[opt.constr_offset_cone[i] + 1]
    end
end
function MOI.get(opt::Optimizer, ::MOI.ConstraintDual, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractVectorSet}
    # vector set
    i = ci.value
    if i <= opt.num_eq_constrs
        # constraint is an equality
        os = opt.constr_offset_eq
        return opt.y[(os[i] + 1):os[i + 1]]
    else
        # constraint is conic
        i -= opt.num_eq_constrs
        os = opt.constr_offset_cone
        return opt.z[(os[i] + 1):os[i + 1]]
    end
end
MOI.get(opt::Optimizer, a::MOI.ConstraintDual, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, a, ci)

function MOI.get(opt::Optimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractScalarSet}
    # scalar set
    i = ci.value
    if i <= opt.num_eq_constrs
        # constraint is an equality
        return opt.constr_prim_eq[opt.constr_offset_eq[i] + 1]
    else
        # constraint is conic
        i -= opt.num_eq_constrs
        return opt.constr_prim_cone[opt.constr_offset_cone[i] + 1]
    end
end
function MOI.get(opt::Optimizer, ::MOI.ConstraintPrimal, ci::MOI.ConstraintIndex{F, S}) where {F <: MOI.AbstractFunction, S <: MOI.AbstractVectorSet}
    # vector set
    i = ci.value
    if i <= opt.num_eq_constrs
        # constraint is an equality
        os = opt.constr_offset_eq
        return opt.constr_prim_eq[(os[i] + 1):os[i + 1]]
    else
        # constraint is conic
        i -= opt.num_eq_constrs
        os = opt.constr_offset_cone
        return opt.constr_prim_cone[(os[i] + 1):os[i + 1]]
    end
end
MOI.get(opt::Optimizer, a::MOI.ConstraintPrimal, ci::Vector{MOI.ConstraintIndex}) = MOI.get.(opt, a, ci)
