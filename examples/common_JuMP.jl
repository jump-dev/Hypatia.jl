#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for JuMP examples
=#

include(joinpath(@__DIR__, "common.jl"))

import JuMP
const MOI = JuMP.MOI

# SOCone, PSDCone, ExpCone, PowerCone only
MOI.Utilities.@model(StandardConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.PositiveSemidefiniteConeTriangle, MOI.ExponentialCone,),
    (MOI.PowerCone, MOI.DualPowerCone,),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

# SOCone and PSDCone only
MOI.Utilities.@model(SOPSDConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone,
    MOI.PositiveSemidefiniteConeTriangle,),
    (),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )

# ExpCone and PSDCone only
MOI.Utilities.@model(ExpPSDConeOptimizer,
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
function test_extra(inst::ExampleInstanceJuMP{T}, model::JuMP.Model) where T
    @test JuMP.termination_status(model) == MOI.OPTIMAL
end

# run a JuMP instance with a given solver and return solve info
function test(
    E::Type{<:ExampleInstanceJuMP{Float64}}, # an instance of a JuMP example # TODO support generic reals
    inst_data::Tuple,
    extender = nothing, # MOI.Utilities.@model-defined optimizer with subset of cones if using extended formulation
    solver_options = (), # additional non-default solver options specific to the example
    solver_type = Hypatia.Optimizer;
    default_solver_options = (), # default solver options
    test::Bool = true,
    rseed::Int = 1,
    )
    # setup instance and model
    Random.seed!(rseed)
    inst = E(inst_data...)
    build_time = @elapsed model = build(inst)

    # solve and process result
    solver = solver_type(; default_solver_options..., solver_options...)
    result = solve_process(inst, model, solver, extender, test)

    return (extender, build_time, result)
end

function solve_process(
    inst,
    model,
    solver::Hypatia.Optimizer,
    extender,
    test::Bool,
    )
    # setup Hypatia model
    opt = hyp_opt = solver
    if !isnothing(extender)
        # use MOI automated extended formulation
        opt = MOI.Utilities.CachingOptimizer(extender{Float64}(), hyp_opt)
    end
    JuMP.set_optimizer(model, () -> opt)
    JuMP.optimize!(model)

    if test
        # run tests for the example
        test_extra(inst, model)
    end

    # use process result
    return process_result(hyp_opt.model, hyp_opt.solver)
end

function solve_process(
    inst,
    model,
    solver,
    extender,
    test::Bool,
    )
    # setup Hypatia model
    opt = hyp_opt = Hypatia.Optimizer()
    if !isnothing(extender)
        # use MOI automated extended formulation
        opt = MOI.Bridges.full_bridge_optimizer(MOI.Utilities.CachingOptimizer(extender{Float64}(), hyp_opt), Float64)
    end
    backend = JuMP.backend(model)
    MOI.Utilities.reset_optimizer(backend, opt)
    MOI.Utilities.attach_optimizer(backend)
    MOI.Utilities.attach_optimizer(backend.optimizer.model)

    hyp_model = hyp_opt.model
    (A, b, c, G, h) = (hyp_model.A, hyp_model.b, hyp_model.c, hyp_model.G, hyp_model.h)
    (cones, cone_idxs) = (hyp_model.cones, hyp_model.cone_idxs)

    jump_model = JuMP.Model()
    JuMP.@variable(jump_model, x_var[1:length(hyp_model.c)])
    JuMP.@objective(jump_model, Min, dot(c, x_var))
    eq_refs = JuMP.@constraint(jump_model, hyp_model.A * x_var .== hyp_model.b)
    cone_refs = Vector{JuMP.ConstraintRef}(undef, length(hyp_model.cones))
    for (k, cone_k) in enumerate(hyp_model.cones)
        idxs = hyp_model.cone_idxs[k]
        h_k = h[idxs]
        G_k = G[idxs, :]
        moi_set = cone_from_hyp(cone_k)
        if Hypatia.needs_untransform(moi_set)
            Hypatia.untransform_affine(moi_set, h_k)
            for j in 1:size(G_k, 2)
                @views Hypatia.untransform_affine(moi_set, G_k[:, j])
            end
        end
        cone_refs[k] = JuMP.@constraint(jump_model, h_k - G_k * x_var in moi_set)
    end

    opt = solver
    JuMP.set_optimizer(jump_model, () -> opt)
    JuMP.optimize!(jump_model)

    solve_time = JuMP.solve_time(jump_model)
    num_iters = MOI.get(jump_model, MOI.BarrierIterations())
    primal_obj = JuMP.objective_value(jump_model)
    dual_obj = JuMP.dual_objective_value(jump_model)
    moi_status = MOI.get(jump_model, MOI.TerminationStatus())
    hyp_status = haskey(moi_hyp_status_map, moi_status) ? moi_hyp_status_map[moi_status] : :OtherStatus

    x = JuMP.value.(x_var)
    y = (isempty(A) ? Float64[] : -JuMP.dual.(eq_refs))
    s_cones = Vector{Vector{Float64}}(undef, length(cone_refs))
    z_cones = Vector{Vector{Float64}}(undef, length(cone_refs))
    for (k, cr) in enumerate(cone_refs)
        moi_set = MOI.get(cr.model, MOI.ConstraintSet(), cr)
        idxs = Hypatia.permute_affine(moi_set, 1:length(hyp_model.cone_idxs[k]))
        s_k = Hypatia.rescale_affine(moi_set, JuMP.value.(cr))
        z_k = Hypatia.rescale_affine(moi_set, JuMP.dual.(cr))
        s_cones[k] = s_k[idxs]
        z_cones[k] = z_k[idxs]
    end
    s = vcat(s_cones...)
    z = vcat(z_cones...)

    obj_diff = primal_obj - dual_obj
    compl = dot(s, z)
    (x_viol, y_viol, z_viol) = certificate_violations(hyp_status, hyp_model, x, y, z, s)
    string_cones = [string(nameof(c)) for c in unique(typeof.(hyp_model.cones))]

    return (status = hyp_status,
        solve_time = solve_time, num_iters = num_iters,
        primal_obj = primal_obj, dual_obj = dual_obj,
        n = hyp_model.n, p = hyp_model.p, q = hyp_model.q,
        cones = string_cones,
        obj_diff = obj_diff, compl = compl,
        x_viol = x_viol, y_viol = y_viol, z_viol = z_viol)
end


# run a CBF instance with a given solver and return solve info
function test(
    inst::String, # a CBF file name
    solver_options = (), # additional non-default solver options specific to the example
    solver_type = Hypatia.Optimizer,
    )
    cbf_file = joinpath(cblib_dir, inst * ".cbf.gz")
    model = JuMP.read_from_file(cbf_file)

    # delete integer constraints
    int_cons = JuMP.all_constraints(model, JuMP.VariableRef, MOI.Integer)
    JuMP.delete.(model, int_cons)

    opt = solver_type(; solver_options...)
    JuMP.set_optimizer(model, () -> opt)
    JuMP.optimize!(model)

    @test JuMP.termination_status(model) == MOI.OPTIMAL # TODO some may be infeasible

    return process_result(model)
end

moi_hyp_status_map = Dict(
    MOI.OPTIMAL => :Optimal,
    MOI.INFEASIBLE => :PrimalInfeasible,
    MOI.DUAL_INFEASIBLE => :DualInfeasible,
    )

cone_from_hyp(cone::Cones.Cone) = error("cannot transform a Hypatia cone of type $(typeof(cone)) to an MOI cone")
cone_from_hyp(cone::Cones.Nonnegative) = MOI.Nonnegatives(Cones.dimension(cone))
cone_from_hyp(cone::Cones.EpiNormInf) = (Cones.use_dual_barrier(cone) ? MOI.NormOneCone : MOI.NormInfinityCone)(Cones.dimension(cone))
cone_from_hyp(cone::Cones.EpiNormEucl) = MOI.SecondOrderCone(Cones.dimension(cone))
cone_from_hyp(cone::Cones.EpiPerSquare) = MOI.RotatedSecondOrderCone(Cones.dimension(cone))
cone_from_hyp(cone::Cones.HypoPerLog) = (@assert Cones.dimension(cone) == 3; MOI.ExponentialCone())
cone_from_hyp(cone::Cones.EpiSumPerEntropy) = MOI.RelativeEntropyCone(Cones.dimension(cone))
cone_from_hyp(cone::Cones.HypoGeoMean) = MOI.GeometricMeanCone(Cones.dimension(cone))
cone_from_hyp(cone::Cones.Power) = (@assert Cones.dimension(cone) == 3; MOI.PowerCone{Float64}(cone.alpha[1]))
cone_from_hyp(cone::Cones.EpiNormSpectral) = (Cones.use_dual_barrier(cone) ? MOI.NormNuclearCone : MOI.NormSpectralCone)(cone.n, cone.m)
cone_from_hyp(cone::Cones.PosSemidefTri{T, R}) where {R <: Hypatia.RealOrComplex{T}} where {T <: Real} = MOI.PositiveSemidefiniteConeTriangle(cone.side)
cone_from_hyp(cone::Cones.LinMatrixIneq{T}) where {T <: Real} = Hypatia.LinMatrixIneqCone{T}(cone.As)
cone_from_hyp(cone::Cones.HypoPerLogdetTri) = MOI.LogDetConeTriangle(cone.side)
cone_from_hyp(cone::Cones.HypoRootdetTri) = MOI.RootDetConeTriangle(cone.side)
cone_from_hyp(cone::Cones.MatrixEpiPerSquare{T, R}) where {R <: Hypatia.RealOrComplex{T}} where {T <: Real} = Hypatia.MatrixEpiPerSquareCone{T, R}(cone.n, cone.m)
