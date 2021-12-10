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

    hyp_opt = if solver_type == Hypatia.Optimizer
        Hypatia.Optimizer(; solver_options...)
    else
        Hypatia.Optimizer()
    end

    extT = if isnothing(extender)
        MOIU.UniversalFallback(MOIU.Model{Float64}())
    else
        @eval $extender{Float64}()
    end
    opt = MOI.Bridges.full_bridge_optimizer(
        MOIU.CachingOptimizer(extT, hyp_opt), Float64)
    if !isnothing(extender)
        # for PolyJuMP/SumOfSquares models
        for B in model.bridge_types
            MOI.Bridges.add_bridge(opt, B{Float64})
        end
    end

    backend = JuMP.backend(model)
    MOIU.reset_optimizer(backend, opt)
    MOIU.attach_optimizer(backend)
    MOIU.attach_optimizer(backend.optimizer.model)
    flush(stdout); flush(stderr)

    hyp_model = hyp_opt.solver.orig_model
    if solver_type == Hypatia.Optimizer
        model.ext[:inst] = inst
    else
        # not using Hypatia to solve, so setup new JuMP model with Hypatia data
        model = JuMP.Model()
        c = hyp_model.c
        JuMP.@variable(model, x[1:length(c)])
        JuMP.@objective(model, Min, dot(c, x) + hyp_model.obj_offset)
        eq_refs = JuMP.@constraint(model, hyp_model.A * x .== hyp_model.b)
        cone_refs = JuMP.ConstraintRef[]
        for (cone, idxs) in zip(hyp_opt.moi_cones, hyp_opt.moi_cone_idxs)
            h_i = hyp_model.h[idxs]
            G_i = hyp_model.G[idxs, :]
            if Hypatia.needs_untransform(cone)
                Hypatia.untransform_affine(cone, h_i)
                @inbounds @views for j in 1:size(G_i, 2)
                    Hypatia.untransform_affine(cone, G_i[:, j])
                end
            end
            con_i = JuMP.@constraint(model, h_i - G_i * x in cone)
            push!(cone_refs, con_i)
        end

        model.ext[:moi_cones] = hyp_opt.moi_cones
        model.ext[:hyp_model] = hyp_model
        model.ext[:x_var] = x
        model.ext[:eq_refs] = eq_refs
        model.ext[:cone_refs] = cone_refs

        JuMP.set_optimizer(model, solver_type)
        for (option, value) in pairs(solver_options)
            JuMP.set_optimizer_attribute(model, string(option), value)
        end
    end
    flush(stdout); flush(stderr)

    return (model, get_model_stats(hyp_model))
end

function solve_check(
    model::JuMP.Model;
    test::Bool = true,
    )
    JuMP.optimize!(model) # TODO make sure it doesn't copy again
    flush(stdout); flush(stderr)

    if JuMP.solver_name(model) == "Hypatia"
        solver = JuMP.backend(model).optimizer.model.optimizer.solver
        test && test_extra(model.ext[:inst], model)
        flush(stdout); flush(stderr)
        (solve_stats, _) = process_result(solver.orig_model, solver)
        return solve_stats
    elseif test
        @info("cannot run example tests if solver is not Hypatia")
    end

    solve_time = JuMP.solve_time(model)
    iters = MOI.get(model, MOI.BarrierIterations())
    primal_obj = JuMP.objective_value(model)
    dual_obj = JuMP.dual_objective_value(model)
    moi_status = JuMP.termination_status(model)
    if haskey(moi_hyp_status_map, moi_status)
        hyp_status = moi_hyp_status_map[moi_status]
    else
        @warn("MOI status $moi_status not handled")
        hyp_status = Solvers.UnknownStatus
    end

    x = JuMP.value.(model.ext[:x_var])
    eq_refs = model.ext[:eq_refs]
    y = (isempty(eq_refs) ? Float64[] : -JuMP.dual.(eq_refs))
    s = Float64[]
    z = Float64[]
    for (cone, cr) in zip(model.ext[:moi_cones], model.ext[:cone_refs])
        s_k = JuMP.value.(cr)
        z_k = JuMP.dual.(cr)
        if Hypatia.needs_rescale(cone)
            Hypatia.rescale_affine(cone, s_k)
            Hypatia.rescale_affine(cone, z_k)
        end
        if Hypatia.needs_permute(cone)
            Hypatia.permute_affine(cone, s_k)
            Hypatia.permute_affine(cone, z_k)
        end
        append!(s, s_k)
        append!(z, z_k)
    end

    hyp_model = model.ext[:hyp_model]
    (x_viol, y_viol, z_viol, compl, rel_obj_diff) =
        certificate_violations(hyp_status, primal_obj, dual_obj, hyp_model, x, y, z, s)
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
    MOI.ALMOST_OPTIMAL => Solvers.NearOptimal,
    MOI.INFEASIBLE => Solvers.PrimalInfeasible,
    MOI.ALMOST_INFEASIBLE => Solvers.NearPrimalInfeasible,
    MOI.DUAL_INFEASIBLE => Solvers.DualInfeasible,
    MOI.ALMOST_DUAL_INFEASIBLE => Solvers.NearDualInfeasible,
    MOI.SLOW_PROGRESS => Solvers.SlowProgress,
    MOI.ITERATION_LIMIT => Solvers.IterationLimit,
    MOI.TIME_LIMIT => Solvers.TimeLimit,
    MOI.NUMERICAL_ERROR => Solvers.NumericalFailure,
    MOI.OTHER_ERROR => Solvers.UnknownStatus,
    )
