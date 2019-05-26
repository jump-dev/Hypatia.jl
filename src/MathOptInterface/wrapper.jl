#=
Copyright 2018, Chris Coey and contributors

MathOptInterface wrapper of Hypatia solver
=#

mutable struct Optimizer <: MOI.AbstractOptimizer
    use_dense::Bool
    test_certificates::Bool
    verbose::Bool
    system_solver::Type{<:Solvers.CombinedHSDSystemSolver}
    linear_model::Type{<:Models.LinearModel}
    tol_rel_opt::Float64
    tol_abs_opt::Float64
    tol_feas::Float64
    max_iters::Int
    time_limit::Float64

    c
    A
    b
    G
    h
    cones
    cone_idxs

    solver::Solvers.HSDSolver

    obj_sense::MOI.OptimizationSense
    obj_const::Float64
    num_eq_constrs::Int
    constr_offset_eq::Vector{Int}
    constr_prim_eq::Vector{Float64}
    constr_offset_cone::Vector{Int}
    constr_prim_cone::Vector{Float64}
    interval_idxs::UnitRange{Int}
    interval_scales::Vector{Float64}

    x::Vector{Float64}
    s::Vector{Float64}
    y::Vector{Float64}
    z::Vector{Float64}
    status::Symbol
    solve_time::Float64
    primal_obj::Float64
    dual_obj::Float64

    load_only::Bool

    function Optimizer(use_dense::Bool, test_certificates::Bool, verbose::Bool, system_solver::Type{<:Solvers.CombinedHSDSystemSolver}, linear_model::Type{<:Models.LinearModel}, max_iters::Int, time_limit::Float64, tol_rel_opt::Float64, tol_abs_opt::Float64, tol_feas::Float64, load_only::Bool)
        opt = new()
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
        opt.status = :NotLoaded
        opt.load_only = load_only
        return opt
    end
end

Optimizer(;
    use_dense::Bool = true,
    test_certificates::Bool = false,
    verbose::Bool = false,
    system_solver::Type{<:Solvers.CombinedHSDSystemSolver} = Solvers.QRCholCombinedHSDSystemSolver,
    linear_model::Type{<:Models.LinearModel} = Models.PreprocessedLinearModel,
    max_iters::Int = 500,
    time_limit::Float64 = 3.6e3, # TODO should be Inf
    tol_rel_opt::Float64 = 1e-6,
    tol_abs_opt::Float64 = 1e-7,
    tol_feas::Float64 = 1e-7,
    load_only::Bool = false,
    ) = Optimizer(use_dense, test_certificates, verbose, system_solver, linear_model, max_iters, time_limit, tol_rel_opt, tol_abs_opt, tol_feas, load_only)

MOI.get(::Optimizer, ::MOI.SolverName) = "Hypatia"

MOI.is_empty(opt::Optimizer) = (opt.status == :NotLoaded)
MOI.empty!(opt::Optimizer) = (opt.status = :NotLoaded) # TODO empty the data and results? keep options?

MOI.supports(::Optimizer, ::Union{
    MOI.ObjectiveSense,
    MOI.ObjectiveFunction{MOI.SingleVariable},
    MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}},
    ) = true

# TODO don't restrict to Float64 type
SupportedFuns = Union{
    MOI.SingleVariable, MOI.ScalarAffineFunction{Float64},
    MOI.VectorOfVariables, MOI.VectorAffineFunction{Float64},
    }

SupportedSets = Union{
    MOI.EqualTo{Float64}, MOI.Zeros,
    MOI.GreaterThan{Float64}, MOI.Nonnegatives,
    MOI.LessThan{Float64}, MOI.Nonpositives,
    MOI.Interval{Float64},
    MOIOtherCones...
    }

MOI.supports_constraint(::Optimizer, ::Type{<:SupportedFuns}, ::Type{<:SupportedSets}) = true

# build representation as min c'x s.t. A*x = b, h - G*x in K
function MOI.copy_to(
    opt::Optimizer,
    src::MOI.ModelLike;
    copy_names::Bool = false,
    warn_attributes::Bool = true,
    )
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
        obj = MOI.ScalarAffineFunction{Float64}(MOI.get(src, MOI.ObjectiveFunction{F}()))
    elseif F == MOI.ScalarAffineFunction{Float64}
        obj = MOI.get(src, MOI.ObjectiveFunction{F}())
    end
    (Jc, Vc) = (Int[], Float64[])
    for t in obj.terms
        push!(Jc, idx_map[t.variable_index].value)
        push!(Vc, t.coefficient)
    end
    if MOI.get(src, MOI.ObjectiveSense()) == MOI.MAX_SENSE
        Vc .*= -1.0
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
    (IA, JA, VA) = (Int[], Int[], Float64[])
    (Ib, Vb) = (Int[], Float64[])
    (Icpe, Vcpe) = (Int[], Float64[]) # constraint set constants for opt.constr_prim_eq
    constr_offset_eq = Vector{Int}()

    # TODO can preprocess variables equal to constant
    for ci in get_src_cons(MOI.SingleVariable, MOI.EqualTo{Float64})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.EqualTo{Float64}}(i)
        push!(constr_offset_eq, p)
        p += 1
        push!(IA, p)
        push!(JA, idx_map[get_con_fun(ci).variable].value)
        push!(VA, -1.0)
        push!(Ib, p)
        push!(Vb, -get_con_set(ci).value)
        push!(Icpe, p)
        push!(Vcpe, get_con_set(ci).value)
    end

    for ci in get_src_cons(MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.EqualTo{Float64}}(i)
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

    # TODO can preprocess variables equal to zero here
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

    for ci in get_src_cons(MOI.VectorAffineFunction{Float64}, MOI.Zeros)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Zeros}(i)
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
    model_b = Vector(sparsevec(Ib, Vb, p)) # TODO if type less strongly, this can be sparse too
    opt.num_eq_constrs = i
    opt.constr_prim_eq = Vector(sparsevec(Icpe, Vcpe, p))
    opt.constr_offset_eq = constr_offset_eq

    # conic constraints
    q = 0 # rows of G (cone constraint matrix)
    (IG, JG, VG) = (Int[], Int[], Float64[])
    (Ih, Vh) = (Int[], Float64[])
    (Icpc, Vcpc) = (Int[], Float64[]) # constraint set constants for opt.constr_prim_eq
    constr_offset_cone = Vector{Int}()
    cones = Cones.Cone[]
    cone_idxs = UnitRange{Int}[]

    # build up one nonnegative cone
    nonneg_start = q

    for ci in get_src_cons(MOI.SingleVariable, MOI.GreaterThan{Float64})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.GreaterThan{Float64}}(i)
        push!(constr_offset_cone, q)
        q += 1
        push!(IG, q)
        push!(JG, idx_map[get_con_fun(ci).variable].value)
        push!(VG, -1.0)
        push!(Ih, q)
        push!(Vh, -get_con_set(ci).lower)
        push!(Vcpc, get_con_set(ci).lower)
        push!(Icpc, q)
    end

    for ci in get_src_cons(MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.GreaterThan{Float64}}(i)
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
            push!(VG, -1.0)
        end
    end

    for ci in get_src_cons(MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonnegatives}(i)
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
        push!(cones, Cones.Nonnegative(q - nonneg_start))
        push!(cone_idxs, (nonneg_start + 1):q)
    end

    # build up one nonpositive cone
    nonpos_start = q

    for ci in get_src_cons(MOI.SingleVariable, MOI.LessThan{Float64})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.LessThan{Float64}}(i)
        push!(constr_offset_cone, q)
        q += 1
        push!(IG, q)
        push!(JG, idx_map[get_con_fun(ci).variable].value)
        push!(VG, -1.0)
        push!(Ih, q)
        push!(Vh, -get_con_set(ci).upper)
        push!(Vcpc, get_con_set(ci).upper)
        push!(Icpc, q)
    end

    for ci in get_src_cons(MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.LessThan{Float64}}(i)
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
            push!(VG, -1.0)
        end
    end

    for ci in get_src_cons(MOI.VectorAffineFunction{Float64}, MOI.Nonpositives)
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.VectorAffineFunction{Float64}, MOI.Nonpositives}(i)
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
        push!(cones, Cones.Nonpositive(q - nonpos_start))
        push!(cone_idxs, (nonpos_start + 1):q)
    end

    # build up one L_infinity norm cone from two-sided interval constraints
    interval_start = q
    num_intervals = MOI.get(src, MOI.NumberOfConstraints{MOI.SingleVariable, MOI.Interval{Float64}}()) +
        MOI.get(src, MOI.NumberOfConstraints{MOI.ScalarAffineFunction{Float64}, MOI.Interval{Float64}}())
    interval_scales = Vector{Float64}(undef, num_intervals)

    if num_intervals > 0
        i += 1
        push!(constr_offset_cone, q)
        q += 1
        push!(Ih, q)
        push!(Vh, 1.0)
    end

    interval_count = 0

    for ci in get_src_cons(MOI.SingleVariable, MOI.Interval{Float64})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.SingleVariable, MOI.Interval{Float64}}(i)
        push!(constr_offset_cone, q)
        q += 1

        upper = get_con_set(ci).upper
        lower = get_con_set(ci).lower
        @assert isfinite(upper) && isfinite(lower)
        @assert upper > lower
        mid = 0.5 * (upper + lower)
        scal = 2.0 * inv(upper - lower)

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

    for ci in get_src_cons(MOI.ScalarAffineFunction{Float64}, MOI.Interval{Float64})
        i += 1
        idx_map[ci] = MOI.ConstraintIndex{MOI.ScalarAffineFunction{Float64}, MOI.Interval{Float64}}(i)
        push!(constr_offset_cone, q)
        q += 1

        upper = get_con_set(ci).upper
        lower = get_con_set(ci).lower
        @assert isfinite(upper) && isfinite(lower)
        @assert upper > lower
        mid = 0.5 * (upper + lower)
        scal = 2.0 / (upper - lower)

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
        push!(cones, Cones.EpiNormInf(q - interval_start))
        push!(cone_idxs, (interval_start + 1):q)
    end

    # add non-LP conic constraints

    for S in MOIOtherCones, F in (MOI.VectorOfVariables, MOI.VectorAffineFunction{Float64})
        for ci in get_src_cons(F, S)
            i += 1
            idx_map[ci] = MOI.ConstraintIndex{F, S}(i)
            push!(constr_offset_cone, q)
            fi = get_con_fun(ci)
            si = get_con_set(ci)
            dim = MOI.output_dimension(fi)
            if F == MOI.VectorOfVariables
                append!(JG, idx_map[vj].value for vj in fi.variables)
                (IGi, VGi, conei) = build_var_cone(fi, si, dim, q)
            else
                append!(JG, idx_map[vt.scalar_term.variable_index].value for vt in fi.terms)
                (IGi, VGi, Ihi, Vhi, conei) = build_constr_cone(fi, si, dim, q)
                append!(Ih, Ihi)
                append!(Vh, Vhi)
            end
            append!(IG, IGi)
            append!(VG, VGi)
            push!(cones, conei)
            push!(cone_idxs, (q + 1):(q + dim))
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
    opt.cone_idxs = cone_idxs

    opt.constr_offset_cone = constr_offset_cone
    opt.constr_prim_cone = Vector(sparsevec(Icpc, Vcpc, q))

    opt.status = :Loaded

    return idx_map
end

function MOI.optimize!(opt::Optimizer)
    if opt.load_only
        return
    end
    model = opt.linear_model(copy(opt.c), copy(opt.A), copy(opt.b), copy(opt.G), copy(opt.h), opt.cones, opt.cone_idxs)
    stepper = Solvers.CombinedHSDStepper(model, system_solver = opt.system_solver(model))
    solver = Solvers.HSDSolver(
        model, stepper = stepper,
        verbose = opt.verbose, max_iters = opt.max_iters, time_limit = opt.time_limit,
        tol_rel_opt = opt.tol_rel_opt, tol_abs_opt = opt.tol_abs_opt, tol_feas = opt.tol_feas,
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
    # TODO refac out primitive cone untransformations
    for k in eachindex(opt.cones)
        if opt.cones[k] isa Cones.PosSemidef
            idxs = opt.cone_idxs[k]
            scale_vec = svec_unscale(length(idxs))
            opt.s[idxs] .*= scale_vec
            opt.z[idxs] .*= scale_vec
        elseif opt.cones[k] isa Cones.HypoPerLogdet
            idxs = opt.cone_idxs[k][3:end]
            scale_vec = svec_unscale(length(idxs))
            opt.s[idxs] .*= scale_vec
            opt.z[idxs] .*= scale_vec
        end
    end
    opt.s[opt.interval_idxs] ./= opt.interval_scales
    opt.constr_prim_cone += opt.s
    opt.z[opt.interval_idxs] .*= opt.interval_scales

    return
end

# function MOI.free!(opt::Optimizer) # TODO ?

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
        return MOI.OTHER_RESULT_STATUS # TODO later distinguish primal/dual ill posed certificates
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
        return MOI.OTHER_RESULT_STATUS # TODO later distinguish primal/dual ill posed certificates
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

function MOI.get(opt::Optimizer, ::MOI.ObjectiveBound)
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
