#=
MOI.Test linear and conic tests
=#

using Test
import MathOptInterface
const MOI = MathOptInterface
import Hypatia

function test_moi(T::Type{<:Real}; solver_options...)
    optimizer = MOI.Utilities.CachingOptimizer(
        MOI.Utilities.UniversalFallback(MOI.Utilities.Model{T}()),
        Hypatia.Optimizer{T}(; solver_options...),
    )

    tol = 2 * sqrt(sqrt(eps(T)))
    config = MOI.Test.Config(
        T,
        atol = tol,
        rtol = tol,
        exclude = Any[
            MOI.ConstraintBasisStatus,
            MOI.VariableBasisStatus,
            MOI.ConstraintName,
            MOI.VariableName,
            MOI.ObjectiveBound,
        ],
    )

    excludes = String[
        # not implemented:
        "test_attribute_SolverVersion",
        # TODO investigate why these run at all:
        "Indicator", 
        "Integer",
        "ZeroOne",
        # TODO fix:
        "test_model_copy_to_Unsupported",
        "test_solve_ObjectiveBound_MAX_SENSE_LP",
        "test_unbounded",
        "test_solve_result_index",
    ]
    includes = String[]

    MOI.Test.runtests(optimizer, config, include = includes, exclude = excludes)

    return
end
