#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for JuMP examples
=#

include(joinpath(@__DIR__, "common.jl"))

import JuMP
const MOI = JuMP.MOI

# SOCone, PSDCone, ExpCone, PowerCone only
MOI.Utilities.@model(ClassicConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.PositiveSemidefiniteConeTriangle, MOI.ExponentialCone, MOI.DualExponentialCone,),
    (MOI.PowerCone, MOI.DualPowerCone,),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

# ExpCone only
MOI.Utilities.@model(ExpConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.ExponentialCone,),
    (),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

# SOCone only
MOI.Utilities.@model(SOConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,),
    (),
    (MOI.SingleVariable,),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

abstract type ExampleInstanceJuMP{T <: Real} <: ExampleInstance{T} end

# fallback: just check optimal status
function test_extra(inst::ExampleInstanceJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

# run a JuMP instance with a given solver and return solve info
function test(
    E::Type{<:ExampleInstanceJuMP{Float64}}, # an instance of a JuMP example # TODO support generic reals
    inst_data::Tuple,
    extender = nothing, # MOI.Utilities.@model-defined optimizer with subset of cones if using extended formulation
    solver_options = (),
    solver::Type{<:MOI.AbstractOptimizer} = Hypatia.Optimizer; # additional non-default solver options specific to the example
    default_solver_options = (), # default solver options
    process_extended_certificates::Bool = false, # TODO default to true # whether to process the certificates for the extended space model (for Hypatia only) or the natural space model
    rseed::Int = 1,
    )
    # setup instance and model
    Random.seed!(rseed)
    inst = E(inst_data...)
    build_time = @elapsed model = build(inst)
    model_backend = JuMP.backend(model)

    # solve
    opt = solver{Float64}(; default_solver_options..., solver_options...)
    if isnothing(extender)
        # solve without MOI extended formulation
        JuMP.set_optimizer(model, () -> opt)
        JuMP.optimize!(model)
    else
        # solve with MOI automated extended formulation
        JuMP.set_optimizer(model, extender{Float64})
        MOI.Utilities.attach_optimizer(model_backend)
        MOI.copy_to(opt, model_backend.optimizer.model)
        JuMP.set_optimizer(model, () -> opt)
        MOI.optimize!(opt)
    end

    # run tests for the example
    test_extra(inst, model)

    # process the solve info and solution
    # TODO any problem with using the isnothing(extender) in the if statement?
    hypatia_opt = model_backend.optimizer.model.optimizer
    if isnothing(extender) || (process_extended_certificates && solver <: Hypatia.Optimizer)
        # use native process result function to calculate residuals on extended certificates stored inside the Hypatia optimizer struct
        result = process_result(hypatia_opt.model, hypatia_opt.solver)
    else
        result = process_result_JuMP(model, hypatia_opt)
    end

    return (extender, build_time, result)
end

# return solve information and certificate violations
function process_result_JuMP(model::JuMP.Model, hypatia_opt::Hypatia.Optimizer{T}) where {T <: Real}
    status = JuMP.termination_status(model)
    solve_time = JuMP.solve_time(model)
    num_iters = MOI.get(model, MOI.BarrierIterations())
    primal_obj = JuMP.objective_value(model)
    dual_obj = JuMP.dual_objective_value(model)

    # get native data in natural space from MOI.copy_to without extension
    # TODO

    # get native certificates in natural space
    # TODO fix old code
    # x = JuMP.value.(x)
    # y = (isempty(A) ? Float64[] : -JuMP.dual.(lin_ref))
    # z = vcat([JuMP.dual.(c) for c in conic_refs]...)
    # s = vcat([JuMP.value.(c) for c in conic_refs]...)
    # transform_moi_convention(G, h, s, z, cones, cone_idxs, opt)
    # kkt_data = get_kkt(x, s, y, z, A, b, c, G, h, flip_sense, tol)

    # process certificates
    obj_diff = primal_obj - dual_obj
    compl = dot(s, z)
    (x_viol, y_viol, z_viol) = certificate_violations(status, hypatia_model, x, y, z, s)

    return (status = status,
        solve_time = solve_time, num_iters = num_iters,
        primal_obj = primal_obj, dual_obj = dual_obj,
        x = x, y = y, s = s, z = z,
        obj_diff = obj_diff, compl = compl,
        x_viol = x_viol, y_viol = y_viol, z_viol = z_viol)
end
