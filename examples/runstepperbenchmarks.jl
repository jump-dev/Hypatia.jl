#=
run stepper benchmarks from the examples folder
unlike runbenchmarks.jl this script does not spawn and is meant for benchmarking Hypatia steppers only
to use the "various" instance set, uncomment it in testaux.jl and run on cmd line:
killall julia; ~/julia/julia examples/runstepperbenchmarks.jl &> ~/bench/bench.txt
=#

examples_dir = joinpath(@__DIR__, "../examples")
(perf, test_instances) = include(joinpath(examples_dir, "testaux.jl"))

# path to write results DataFrame to CSV, if any
results_path = joinpath(homedir(), "bench", "bench.csv")
# results_path = nothing

# default options to solvers
default_options = (
    # verbose = false,
    verbose = true,
    default_tol_relax = 10,
    stepper = Solvers.CombinedStepper{Float64}(),
    # stepper = Solvers.PredOrCentStepper{Float64}(),
    iter_limit = 1000,
    )

    # instance sets and real types to run and corresponding time limits (seconds)
    instance_sets = [
        ("minimal", Float64, 60),
        # ("minimal", Float32, 60),
        # ("minimal", BigFloat, 60),
        # ("fast", Float64, 60),
        # ("slow", Float64, 120),
        ]

# types of models to run and corresponding options and example names
model_types = [
    "native",
    "JuMP",
    ]

# list of names of native examples to run
native_example_names = [
    "densityest",
    # "envelope",
    # "expdesign",
    # "linearopt",
    # "matrixcompletion",
    # "matrixregression",
    # "maxvolume",
    # "polymin",
    # "portfolio",
    # "sparsepca",
    ]

# list of names of JuMP examples to run
JuMP_example_names = [
    # "centralpolymat",
    # "conditionnum",
    # "contraction",
    # "densityest",
    # "envelope",
    # "expdesign",
    # "lotkavolterra",
    # "lyapunovstability",
    # "matrixcompletion",
    # "matrixquadratic",
    # "matrixregression",
    # "maxvolume",
    # "muconvexity",
    # "nearestpsd",
    # "polymin",
    # "polynorm",
    # "portfolio",
    # "nearestpolymat",
    # "regionofattr",
    # "robustgeomprog",
    # "secondorderpoly",
    # "semidefinitepoly",
    # "shapeconregr",
    # "signomialmin",
    # "stabilitynumber",
    ]

isnothing(results_path) || CSV.write(results_path, perf)

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
        test_instances(mod_type, steppers, instance_sets, ex_insts, ex_type, default_options, results_path)
    end
end
end
