#=
MOI.Test linear and conic tests
=#

using Test
import MathOptInterface
const MOI = MathOptInterface
import Hypatia

function test_moi(T::Type{<:Real}; solver_options...)
    MOI.Test.runtests(
        MOI.Utilities.CachingOptimizer(
            MOI.Utilities.UniversalFallback(MOI.Utilities.Model{T}()),
            Hypatia.Optimizer{T}(; solver_options...),
        ),
        MOI.Test.Config(
            T,
            atol = 2 * sqrt(sqrt(eps(T))),
            rtol = 2 * sqrt(sqrt(eps(T))),
            exclude = Any[
                MOI.ConstraintBasisStatus,
                MOI.VariableBasisStatus,
                MOI.ObjectiveBound,
            ],
        ),
        exclude = String[
            # not implemented:
            "test_attribute_SolverVersion",
            # TODO fix:
            "test_model_copy_to_Unsupported",
        ],
    )

    return
end
