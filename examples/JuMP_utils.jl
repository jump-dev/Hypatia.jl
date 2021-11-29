#=
common code for JuMP examples
=#

import JuMP
const MOI = JuMP.MOI
const MOIU = MOI.Utilities

MOIU.@model(SOCExpPSD,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,
    MOI.ExponentialCone, MOI.PositiveSemidefiniteConeTriangle,),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

MOIU.@model(ExpPSD,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.ExponentialCone, MOI.PositiveSemidefiniteConeTriangle,),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

abstract type ExampleInstanceJuMP{T <: Real} <: ExampleInstance{T} end

# fallback: just check optimal status
function test_extra(inst::ExampleInstanceJuMP, model::JuMP.Model)
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

function run_instance(
    ex_type::Type{<:ExampleInstanceJuMP},
    inst_data::Tuple,
    extender::Union{Symbol, Nothing} = nothing,
    inst_options::NamedTuple = NamedTuple(),
    solver_type = Hypatia.Optimizer;
    default_options::NamedTuple = NamedTuple(),
    test::Bool = true,
    verbose::Bool = true,
    )
    new_options = merge(default_options, inst_options)

    verbose && println("setup model")
    setup_time = @elapsed (model, model_stats) =
        setup_model(ex_type, inst_data, extender, new_options, solver_type)

    verbose && println("solve and check")
    check_time = @elapsed solve_stats = solve_check(model, test = test)

    return (; model_stats..., solve_stats..., setup_time,
        check_time, :script_status => "Success")
end

function setup_model(
    ex_type::Type{<:ExampleInstanceJuMP},
    inst_data::Tuple,
    extender::Union{Symbol, Nothing},
    solver_options::NamedTuple,
    solver_type;
    rseed::Int = 1,
    )
    # setup example instance and JuMP model
    Random.seed!(rseed)
    inst = ex_type(inst_data...)
    model = build(inst)

    is_hypatia_opt = (solver_type == Hypatia.Optimizer)
    opt = hyp_opt = (is_hypatia_opt ? Hypatia.Optimizer(; solver_options...) :
        Hypatia.Optimizer(use_dense_model = false))
    if !isnothing(extender)
        # use MOI automated extended formulation
        extT = @eval $extender{Float64}()
        opt = MOI.Bridges.full_bridge_optimizer(
            MOIU.CachingOptimizer(extT, opt), Float64)
        # for PolyJuMP/SumOfSquares models
        for B in model.bridge_types
            MOI.Bridges.add_bridge(opt, B{Float64})
        end
    end
    backend = JuMP.backend(model)
    MOIU.reset_optimizer(backend, opt)
    MOIU.attach_optimizer(backend)
    isnothing(extender) || MOIU.attach_optimizer(backend.optimizer.model)
    flush(stdout); flush(stderr)

    hyp_model = hyp_opt.model
    if is_hypatia_opt
        model.ext[:inst] = inst
    else
        # not using Hypatia to solve, so setup new JuMP model with Hypatia data
        (A, b, c, G, h) =
            (hyp_model.A, hyp_model.b, hyp_model.c, hyp_model.G, hyp_model.h)
        (cones, cone_idxs) = (hyp_model.cones, hyp_model.cone_idxs)
        new_model = JuMP.Model()
        new_model.ext[:hyp_data] = hyp_model
        JuMP.@variable(new_model, x[1:length(c)])
        JuMP.@objective(new_model, Min, dot(c, x))
        eq_refs = JuMP.@constraint(new_model, A * x .== b)
        cone_refs = Vector{JuMP.ConstraintRef}(undef, length(cones))
        for (k, cone_k) in enumerate(cones)
            idxs = cone_idxs[k]
            h_k = h[idxs]
            G_k = G[idxs, :]
            moi_set = cone_from_hyp(cone_k)
            if Hypatia.needs_untransform(moi_set)
                Hypatia.untransform_affine(moi_set, h_k)
                @inbounds @views for j in 1:size(G_k, 2)
                    Hypatia.untransform_affine(moi_set, G_k[:, j])
                end
            end
            cone_refs[k] = JuMP.@constraint(new_model, h_k - G_k * x in moi_set)
        end
        new_model.ext[:x_var] = x
        new_model.ext[:eq_refs] = eq_refs
        new_model.ext[:cone_refs] = cone_refs

        opt = solver_type(; solver_options...)
        JuMP.set_optimizer(new_model, () -> opt)
        model = new_model
        flush(stdout); flush(stderr)
    end

    model_stats = get_model_stats(hyp_model)
    return (model, model_stats)
end

function solve_check(
    model::JuMP.Model;
    test::Bool = true,
    )
    JuMP.optimize!(model) # TODO make sure it doesn't copy again
    flush(stdout); flush(stderr)

    opt = JuMP.backend(model).optimizer
    if !isa(opt, Hypatia.Optimizer)
        backend_model = JuMP.backend(model).optimizer.model
        if backend_model isa MOIU.CachingOptimizer
            opt = backend_model.optimizer
        end
    end

    if opt isa Hypatia.Optimizer
        test && test_extra(model.ext[:inst], model)
        flush(stdout); flush(stderr)
        (solve_stats, _) = process_result(opt.model, opt.solver)
        return solve_stats
    end

    test && @info("cannot run example tests if solver is not Hypatia")

    solve_time = JuMP.solve_time(model)
    iters = MOI.get(model, MOI.BarrierIterations())
    primal_obj = JuMP.objective_value(model)
    dual_obj = JuMP.dual_objective_value(model)
    moi_status = MOI.get(model, MOI.TerminationStatus())
    if haskey(moi_hyp_status_map, moi_status)
        hyp_status = moi_hyp_status_map[moi_status]
    else
        @warn("MOI status $moi_status not handled")
        hyp_status = Solvers.UnknownStatus
    end

    hyp_data = model.ext[:hyp_data]
    eq_refs = model.ext[:eq_refs]
    cone_refs = model.ext[:cone_refs]
    x = JuMP.value.(model.ext[:x_var])
    y = (isempty(eq_refs) ? Float64[] : -JuMP.dual.(eq_refs))
    s_cones = Vector{Vector{Float64}}(undef, length(cone_refs))
    z_cones = Vector{Vector{Float64}}(undef, length(cone_refs))
    for (k, cr) in enumerate(cone_refs)
        moi_set = MOI.get(cr.model, MOI.ConstraintSet(), cr)
        idxs = Hypatia.permute_affine(moi_set, 1:length(hyp_data.cone_idxs[k]))
        s_k = Hypatia.rescale_affine(moi_set, JuMP.value.(cr))
        z_k = Hypatia.rescale_affine(moi_set, JuMP.dual.(cr))
        s_cones[k] = s_k[idxs]
        z_cones[k] = z_k[idxs]
    end
    s = vcat(s_cones...)
    z = vcat(z_cones...)

    rel_obj_diff = (primal_obj - dual_obj) / (1 + abs(dual_obj))
    compl = dot(s, z)
    (x_viol, y_viol, z_viol) =
        certificate_violations(hyp_status, hyp_data, x, y, z, s)
    flush(stdout); flush(stderr)

    solve_stats = (;
        :status => hyp_status, solve_time, iters, primal_obj, dual_obj,
        rel_obj_diff, compl, x_viol, y_viol, z_viol,
        )
    return solve_stats
end

# get Hypatia status from MOI status
moi_hyp_status_map = Dict(
    MOI.OPTIMAL => Solvers.Optimal,
    MOI.INFEASIBLE => Solvers.PrimalInfeasible,
    MOI.DUAL_INFEASIBLE => Solvers.DualInfeasible,
    MOI.SLOW_PROGRESS => Solvers.SlowProgress,
    MOI.ITERATION_LIMIT => Solvers.IterationLimit,
    MOI.TIME_LIMIT => Solvers.TimeLimit,
    MOI.OTHER_ERROR => Solvers.UnknownStatus,
    )

# get MOI cones (defined in MOI itself, not Hypatia) from some Hypatia cones
cone_from_hyp(cone::Cones.Cone) =
    error("cannot transform Hypatia cone of type $(typeof(cone)) to MOI cone")

cone_from_hyp(cone::Cones.Nonnegative) = MOI.Nonnegatives(Cones.dimension(cone))

cone_from_hyp(cone::Cones.EpiNormInf{T, T}) where {T <: Real} =
    (Cones.use_dual_barrier(cone) ? MOI.NormOneCone :
    MOI.NormInfinityCone)(Cones.dimension(cone))

cone_from_hyp(cone::Cones.EpiNormEucl) =
    MOI.SecondOrderCone(Cones.dimension(cone))

cone_from_hyp(cone::Cones.EpiPerSquare) =
    MOI.RotatedSecondOrderCone(Cones.dimension(cone))

cone_from_hyp(cone::Cones.HypoPerLog) =
    (@assert Cones.dimension(cone) == 3; MOI.ExponentialCone())

cone_from_hyp(cone::Cones.EpiRelEntropy) =
    MOI.RelativeEntropyCone(Cones.dimension(cone))

cone_from_hyp(cone::Cones.HypoGeoMean) =
    MOI.GeometricMeanCone(Cones.dimension(cone))

cone_from_hyp(cone::Cones.GeneralizedPower) =
    (@assert Cones.dimension(cone) == 3; MOI.PowerCone{Float64}(cone.α[1]))

cone_from_hyp(cone::Cones.EpiNormSpectral{T, T}) where {T <: Real} =
    (Cones.use_dual_barrier(cone) ? MOI.NormNuclearCone :
    MOI.NormSpectralCone)(cone.n, cone.m)

cone_from_hyp(cone::Cones.PosSemidefTri{T, T}) where {T <: Real} =
    MOI.PositiveSemidefiniteConeTriangle(cone.side)

cone_from_hyp(cone::Cones.HypoPerLogdetTri{T, T}) where {T <: Real} =
    MOI.LogDetConeTriangle(cone.side)

cone_from_hyp(cone::Cones.HypoRootdetTri{T, T}) where {T <: Real} =
    MOI.RootDetConeTriangle(cone.side)
