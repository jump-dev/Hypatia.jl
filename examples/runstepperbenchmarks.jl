#=
run stepper benchmarks from the examples folder
unlike runbenchmarks.jl this script does not spawn and is meant for benchmarking Hypatia steppers only
to use the "various" instance set, uncomment it in testaux.jl and run on cmd line:
killall julia; ~/julia/julia examples/runstepperbenchmarks.jl &> ~/bench/bench.txt
=#

examples_dir = joinpath(@__DIR__, "../examples")
(model_types, native_example_names, JuMP_example_names, perf, test_instances) = include(joinpath(examples_dir, "testaux.jl"))

predorcent = Solvers.PredOrCentStepper{Float64}
combined = Solvers.CombinedStepper{Float64}
steppers = [
    (predorcent(use_correction = false, use_curve_search = false)),
    (predorcent(use_correction = true, use_curve_search = false)),
    (predorcent(use_correction = true, use_curve_search = true)),
    (combined()),
    (combined(2)),
    ]

@testset "examples tests" begin
@testset "$mod_type" for mod_type in model_types
    @testset "$ex_name" for ex_name in eval(Symbol(mod_type, "_example_names"))
        include(joinpath(examples_dir, ex_name, mod_type * ".jl"))
        (ex_type, ex_insts) = include(joinpath(examples_dir, ex_name, mod_type * "_test.jl"))
        test_instances(steppers, instance_sets, ex_insts, ex_type, perf)
    end
end
end
