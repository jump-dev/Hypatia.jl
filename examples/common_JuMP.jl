#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

common code for JuMP examples

a model function returns a tuple of (JuMP model, test helpers)

an instance consists of a tuple of:
(1) tuple of args to the example model function
(2) boolean for whether to use MOI automatic extend formulation to a `classic' cone formulation (see ClassicConeOptimizer below)
(3) tuple of options for the example test function
(4) tuple of solver options
=#

using Test
import Random
using LinearAlgebra
import Hypatia
const CO = Hypatia.Cones
const MU = Hypatia.ModelUtilities
import JuMP
const MOI = JuMP.MOI

function test_JuMP_instance(
    model_function::Function,
    test_function::Function,
    instance_info::Tuple;
    T::Type{<:Real} = Float64,
    default_solver_options::NamedTuple = NamedTuple(),
    rseed::Int = 1,
    )
    # setup model
    Random.seed!(rseed)
    (model, test_helpers) = model_function(instance_info[1]...)

    # solve model
    hyp_opt = Hypatia.Optimizer(; default_solver_options..., instance_info[4]...)
    if instance_info[2]
        # use MOI automated extended formulation
        JuMP.set_optimizer(model, ClassicConeOptimizer{Float64})
        MOI.Utilities.attach_optimizer(JuMP.backend(model))
        MOI.copy_to(hyp_opt, JuMP.backend(model).optimizer.model)
    end
    JuMP.set_optimizer(model, () -> hyp_opt)
    JuMP.optimize!(model)

    # run tests for the example
    test_function(model, test_helpers, instance_info[3])

    return model
end

MOI.Utilities.@model(ClassicConeOptimizer,
    (),
    (MOI.EqualTo, MOI.GreaterThan, MOI.LessThan, MOI.Interval,),
    (MOI.Reals, MOI.Zeros, MOI.Nonnegatives, MOI.Nonpositives,
    MOI.SecondOrderCone, MOI.RotatedSecondOrderCone, MOI.PositiveSemidefiniteConeTriangle, MOI.ExponentialCone, MOI.DualExponentialCone,),
    (MOI.PowerCone, MOI.DualPowerCone,
    Hypatia.HypoPerLogCone,
    Hypatia.HypoGeomeanCone,
    Hypatia.MatrixEpiPerSquareCone,
    Hypatia.LinMatrixIneqCone,
    Hypatia.PosSemidefTriSparseCone,
    Hypatia.WSOSInterpNonnegativeCone,
    Hypatia.WSOSInterpPosSemidefTriCone,
    Hypatia.WSOSInterpEpiNormEuclCone,
    ),
    (),
    (MOI.ScalarAffineFunction,),
    (MOI.VectorOfVariables,),
    (MOI.VectorAffineFunction,),
    true,
    )
