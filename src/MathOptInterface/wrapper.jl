#=
Copyright 2018, Chris Coey and contributors

MathOptInterface wrapper of Hypatia solver
=#

mutable struct Optimizer{T <: Real} <: MOI.AbstractOptimizer
    load_only::Bool
    use_dense::Bool
    test_certificates::Bool

    verbose::Bool
    system_solver::Type{<:Solvers.CombinedHSDSystemSolver{T}}
    linear_model::Type{<:Models.LinearModel{T}}
    max_iters::Int
    time_limit::Float64
    tol_rel_opt::T
    tol_abs_opt::T
    tol_feas::T
    tol_slow::T

    c::Vector{T}
    A
    b::Vector{T}
    G
    h::Vector{T}
    cones::Vector{Cones.Cone{T}}

    solver::Solvers.HSDSolver

    obj_sense::MOI.OptimizationSense
    obj_const::T
    num_eq_constrs::Int
    constr_offset_eq::Vector{Int}
    constr_prim_eq::Vector{T}
    constr_offset_cone::Vector{Int}
    constr_prim_cone::Vector{T}
    interval_idxs::UnitRange{Int}
    interval_scales::Vector{T}

    x::Vector{T}
    s::Vector{T}
    y::Vector{T}
    z::Vector{T}
    status::Symbol
    solve_time::Float64
    primal_obj::T
    dual_obj::T

    function Optimizer{T}(;
        load_only::Bool = false,
        use_dense::Bool = true,
        test_certificates::Bool = false,
        verbose::Bool = false,
        system_solver::Type{<:Solvers.CombinedHSDSystemSolver{T}} = Solvers.QRCholCombinedHSDSystemSolver{T},
        linear_model::Type{<:Models.LinearModel{T}} = Models.PreprocessedLinearModel{T},
        max_iters::Int = 500,
        time_limit::Real = 3.6e3,
        tol_rel_opt::Real = max(T(1e-12), T(1e-2) * cbrt(eps(T))),
        tol_abs_opt::Real = tol_rel_opt,
        tol_feas::Real = tol_rel_opt,
        tol_slow::Real = T(5e-3),
        ) where {T <: Real}
        opt = new{T}()

        opt.load_only = load_only
        opt.use_dense = use_dense
        opt.test_certificates = test_certificates
        opt.verbose = verbose
        opt.system_solver = system_solver
        opt.linear_model = linear_model
        opt.max_iters = max_iters
        opt.time_limit = time_limit
        opt.tol_rel_opt = tol_rel_opt
        opt.tol_abs_opt = tol_abs_opt
        opt.tol_feas = tol_feas
        opt.tol_slow = tol_slow

        opt.status = :NotLoaded

        return opt
    end
end

MOI.get(::Optimizer, ::MOI.SolverName) = "Hypatia"
MOI.get(opt::Optimizer, ::MOI.RawSolver) = opt.solver

MOI.is_empty(opt::Optimizer) = (opt.status == :NotLoaded)
MOI.empty!(opt::Optimizer) = (opt.status = :NotLoaded)

MOI.supports(::Optimizer{T}, ::Union{
    MOI.ObjectiveSense,
    MOI.ObjectiveFunction{MOI.SingleVariable},
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{T}}},
    ) where {T <: Real} = true

MOI.supports_constraint(::Optimizer{T},
    ::Type{<:Union{MOI.SingleVariable, MOI.ScalarAffineFunction{T}}},
    ::Type{<:Union{MOI.EqualTo{T}, MOI.GreaterThan{T}, MOI.LessThan{T}, MOI.Interval{T}}}
    ) where {T <: Real} = true
MOI.supports_constraint(::Optimizer{T},
    ::Type{<:Union{MOI.VectorOfVariables, MOI.VectorAffineFunction{T}}},
    ::Type{<:Union{MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives, MOIOtherCones{T}}}
    ) where {T <: Real} = true

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.set(opt::Optimizer, ::MOI.Silent, value::Bool) = (opt.verbose = value)
MOI.get(opt::Optimizer, ::MOI.Silent) = opt.verbose

# build representation as min c'x s.t. A*x = b, h - G*x in K
function MOI.copy_to(
    opt::Optimizer{T},
    src::MOI.ModelLike;
    copy_names::Bool = false,
    warn_attributes::Bool = true,
    ) where {T <: Real}
    @assert !copy_names
    idx_map = Dict{MOI.Index, MOI.Index}()

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
    end
    (Jc, Vc) = (Int[], T[])
    for t in obj.terms
        push!(Jc, idx_map[t.variable_index].value)
        push!(Vc, t.coefficient)
    end
    if MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        Vc .*= -1
    end
    opt.obj_const = obj.constant
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
        append!(VA, -ones(dim))
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
    if opt.use_dense
        model_A = Matrix(sparse(IA, JA, VA, p, n))
    else
        model_A = dropzeros!(sparse(IA, JA, VA, p, n))
    end
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

    # build up one nonnegative cone
    nonneg_start = q

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

    if q > nonneg_start
        # exists at least one nonnegative constraint
        push!(cones, Cones.Nonnegative{T}(q - nonneg_start))
    end

    # build up one nonpositive cone
    nonpos_start = q

    for ci in get_src_cons(MOI.SingleVariable, MOI.LessThan{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{T}}(i)
        push!(constr_offset_cone, q)
        q += 1
        push!(IG, q)
        push!(JG, idx_map[get_con_fun(ci).variable].value)
        push!(VG, -1)
        push!(Ih, q)
        push!(Vh, -get_con_set(ci).upper)
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
            push!(VG, -vt.coefficient)
        end
        push!(Ih, q)
        push!(Vh, fi.constant - get_con_set(ci).upper)
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
            push!(VG, -1)
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
            push!(VG, -vt.scalar_term.coefficient)
        end
        append!(Ih, (q + 1):(q + dim))
        append!(Vh, fi.constants)
        q += dim
    end

    if q > nonpos_start
        # exists at least one nonpositive constraint
        push!(cones, Cones.Nonpositive{T}(q - nonpos_start))
    end

    # build up one L_infinity norm cone from two-sided interval constraints
    interval_start = q
    num_intervals = MOI.get(src, MOI.NumberOfConstraints{MOI.SingleVariable, MOI.Interval{T}}()) +
        MOI.get(src, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{T}, MOI.Interval{T}}())
    interval_scales = Vector{T}(undef, num_intervals)

    if num_intervals > 0
        i += 1
        push!(constr_offset_cone, q)
        q += 1
        push!(Ih, q)
        push!(Vh, 1)
    end

    interval_count = 0

    for ci in get_src_cons(MOI.SingleVariable, MOI.Interval{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{T}}(i)
        push!(constr_offset_cone, q)
        q += 1

        upper = get_con_set(ci).upper
        lower = get_con_set(ci).lower
        @assert isfinite(upper) && isfinite(lower)
        @assert upper > lower
        mid = (upper + lower) / 2
        scal = 2 * inv(upper - lower)

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

    for ci in get_src_cons(MOI.ScalarAffineFunction{T}, MOI.Interval{T})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{T}, MOI.Interval{T}}(i)
        push!(constr_offset_cone, q)
        q += 1

        upper = get_con_set(ci).upper
        lower = get_con_set(ci).lower
        @assert isfinite(upper) && isfinite(lower)
        @assert upper > lower
        mid = (upper + lower) / 2
        scal = 2 / (upper - lower)

        fi = get_con_fun(ci)
        for vt in fi.terms
            push!(IG, q)
            push!(JG, idx_map[vt.variable_index].value)
            push!(VG, -vt.coefficient * scal)
        end
        push!(Ih, q)
        push!(Vh, (fi.constant - mid)*scal)
        push!(Vcpc, mid)
        push!(Icpc, q)
        interval_count += 1
        interval_scales[interval_count] = scal
    end

    opt.interval_idxs = (interval_start + 2):q
    opt.interval_scales = interval_scales
    if q > interval_start
        # exists at least one interval-type constraint
        push!(cones, Cones.EpiNormInf{T}(q - interval_start))
    end

    # add non-LP conic constraints

    for S in MOIOtherConesList(T), F in (MOI.VectorOfVariables, MOI.VectorAffineFunction{T})
        for ci in get_src_cons(F, S)
            i += 1
            idx_map[ci] = MOI.ConstraintIndex{F, S}(i)
            push!(constr_offset_cone, q)
            fi = get_con_fun(ci)
            si = get_con_set(ci)
            dim = MOI.output_dimension(fi)
            conei = cone_from_moi(T, si)
            if F == MOI.VectorOfVariables
                append!(JG, idx_map[vj].value for vj in fi.variables)
                IGi = (q + 1):(q + dim)
                VGi = -ones(T, dim)
            else
                append!(JG, idx_map[vt.scalar_term.variable_index].value for vt in fi.terms)
                IGi = [q + vt.output_index for vt in fi.terms]
                VGi = [-vt.scalar_term.coefficient for vt in fi.terms]
                Ihi = (q + 1):(q + dim)
                Vhi = fi.constants
                append!(Ih, Ihi)
                append!(Vh, Vhi)
            end
            append!(IG, IGi)
            append!(VG, VGi)
            push!(cones, conei)
            q += dim
        end
    end

    push!(constr_offset_cone, q)

    if opt.use_dense
        model_G = Matrix(sparse(IG, JG, VG, q, n))
    else
        model_G = dropzeros!(sparse(IG, JG, VG, q, n))
    end
    model_h = Vector(sparsevec(Ih, Vh, q))

    opt.c = model_c
    opt.A = model_A
    opt.b = model_b
    opt.G = model_G
    opt.h = model_h
    opt.cones = cones

    opt.constr_offset_cone = constr_offset_cone
    opt.constr_prim_cone = Vector(sparsevec(Icpc, Vcpc, q))

    opt.status = :Loaded

    return idx_map
end

function MOI.optimize!(opt::Optimizer{T}) where {T <: Real}
    if opt.load_only
        return
    end
    model = opt.linear_model(copy(opt.c), copy(opt.A), copy(opt.b), copy(opt.G), copy(opt.h), opt.cones)
    stepper = Solvers.CombinedHSDStepper{T}(model, system_solver = opt.system_solver(model))
    solver = Solvers.HSDSolver{T}(
        model, stepper = stepper,
        verbose = opt.verbose, max_iters = opt.max_iters, time_limit = opt.time_limit,
        tol_rel_opt = opt.tol_rel_opt, tol_abs_opt = opt.tol_abs_opt, tol_feas = opt.tol_feas, tol_slow = opt.tol_slow,
        )
    Solvers.solve(solver)
    r = Solvers.get_certificates(solver, model, test = opt.test_certificates)

    opt.solve_time = Solvers.get_solve_time(solver)
    opt.status = r.status
    opt.primal_obj = r.primal_obj
    opt.dual_obj = r.dual_obj

    # get solution and transform for MOI
    opt.x = r.x
    opt.constr_prim_eq += opt.b - opt.A * opt.x
    opt.y = r.y
    opt.s = r.s
    opt.z = r.z
    opt.s[opt.interval_idxs] ./= opt.interval_scales
    for (k, cone_k) in enumerate(opt.cones)
        if cone_k isa Cones.PosSemidefTri || cone_k isa Cones.HypoPerLogdetTri # rescale duals for symmetric triangle cones
            cone_idxs_k = Models.get_cone_idxs(model)[k]
            unscale_vec = (Cones.use_dual(cone_k) ? opt.s : opt.z)
            idxs = (cone_k isa Cones.PosSemidefTri ? cone_idxs_k : cone_idxs_k[3:end])
            offset = 1
            for i in 1:round(Int, sqrt(0.25 + 2 * length(idxs)) - 0.5)
                for j in 1:(i - 1)
                    unscale_vec[idxs[offset]] /= 2
                    offset += 1
                end
                offset += 1
            end
        end
    end
    opt.constr_prim_cone += opt.s
    opt.z[opt.interval_idxs] .*= opt.interval_scales

    return
end

MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, value::Real) = (opt.time_limit = value)
MOI.set(opt::Optimizer, ::MOI.TimeLimitSec, ::Nothing) = (opt.time_limit = Inf)
MOI.get(opt::Optimizer, ::MOI.TimeLimitSec) = (isfinite(opt.time_limit) ? opt.time_limit : nothing)

function MOI.get(opt::Optimizer, ::MOI.SolveTime)
    if opt.status in (:NotLoaded, :Loaded)
        error("solve has not been called")
    end
    return opt.solve_time
end

MOI.get(opt::Optimizer, ::MOI.RawStatusString) = string(opt.status)

function MOI.get(opt::Optimizer, ::MOI.TerminationStatus)
    if opt.status in (:NotLoaded, :Loaded)
        return MOI.OPTIMIZE_NOT_CALLED
    elseif opt.status == :Optimal
        return MOI.OPTIMAL
    elseif opt.status == :PrimalInfeasible
        return MOI.INFEASIBLE
    elseif opt.status == :DualInfeasible
        return MOI.DUAL_INFEASIBLE
    elseif opt.status == :SlowProgress
        return MOI.SLOW_PROGRESS
    elseif opt.status == :IterationLimit
        return MOI.ITERATION_LIMIT
    elseif opt.status == :TimeLimit
        return MOI.TIME_LIMIT
    else
        @warn("Hypatia status $(opt.status) not handled")
        return MOI.OTHER_ERROR
    end
end

function MOI.get(opt::Optimizer, ::MOI.PrimalStatus)
    if opt.status == :Optimal
        return MOI.FEASIBLE_POINT
    elseif opt.status == :PrimalInfeasible
        return MOI.INFEASIBLE_POINT
    elseif opt.status == :DualInfeasible
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif opt.status == :IllPosed
        return MOI.OTHER_RESULT_STATUS
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(opt::Optimizer, ::MOI.DualStatus)
    if opt.status == :Optimal
        return MOI.FEASIBLE_POINT
    elseif opt.status == :PrimalInfeasible
        return MOI.INFEASIBILITY_CERTIFICATE
    elseif opt.status == :DualInfeasible
        return MOI.INFEASIBLE_POINT
    elseif opt.status == :IllPosed
        return MOI.OTHER_RESULT_STATUS
    else
        return MOI.UNKNOWN_RESULT_STATUS
    end
end

function MOI.get(opt::Optimizer, ::MOI.ObjectiveValue)
    if opt.obj_sense == MOI.MIN_SENSE
        return opt.primal_obj + opt.obj_const
    elseif opt.obj_sense == MOI.MAX_SENSE
        return -opt.primal_obj + opt.obj_const
    else
        error("no objective sense is set")
    end
end

function MOI.get(opt::Optimizer, ::Union{MOI.DualObjectiveValue, MOI.ObjectiveBound})
    if opt.obj_sense == MOI.MIN_SENSE
        return opt.dual_obj + opt.obj_const
    elseif opt.obj_sense == MOI.MAX_SENSE
        return -opt.dual_obj + opt.obj_const
    else
        error("no objective sense is set")
    end
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
