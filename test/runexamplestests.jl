#=
run examples tests from the examples folder
=#

examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(@__DIR__, "../benchmarks/setup.jl"))

# path to write results DataFrame to CSV, if any
# results_path = joinpath(homedir(), "bench", "bench.csv")
results_path = nothing

# script verbosity
script_verbose = false

# default options to solvers
default_options = (
    verbose = false,
    # verbose = true,
    default_tol_relax = 10,
    # stepper = Solvers.CombinedStepper{Float64}(),
    # stepper = Solvers.PredOrCentStepper{Float64}(),
    iter_limit = 250,
    )

# instance sets and real types to run and corresponding time limits (seconds)
instance_sets = [
    ("minimal", Float64, 60),
    # ("minimal", Float32, 60),
    # ("minimal", BigFloat, 60),
    # ("fast", Float64, 60),
    # ("slow", Float64, 120),
    # ("various", Float64, 120),
    ]

# types of models to run and corresponding options and example names
model_types = [
    "native",
    "JuMP",
    ]

# list of names of native examples to run
native_example_names = [
    "densityest",
    "envelope",
    "expdesign",
    "linearopt",
    "matrixcompletion",
    "matrixregression",
    "maxvolume",
    "polymin",
    "portfolio",
    "sparsepca",
    ]

# list of names of JuMP examples to run
JuMP_example_names = [
    "CBLIB",
    "centralpolymat",
    "classicalquantum",
    "conditionnum",
    "contraction",
    "covarianceest",
    "densityest",
    "entanglementassisted",
    "envelope",
    "expdesign",
    "lotkavolterra",
    "lyapunovstability",
    "matrixcompletion",
    "matrixquadratic",
    "matrixregression",
    "maxvolume",
    "muconvexity",
    "nearestpsd",
    "normconepoly",
    "polymin",
    "polynorm",
    "portfolio",
    "nearestcorrelation",
    "nearestpolymat",
    "nonparametricdistr",
    "regionofattr",
    "relentrentanglement",
    "robustgeomprog",
    "semidefinitepoly",
    "shapeconregr",
    "signomialmin",
    "sparselmi",
    "stabilitynumber",
    ]

perf = setup_benchmark_dataframe()
isnothing(results_path) || CSV.write(results_path, perf)

@testset "examples tests" begin
@testset "$mod_type" for mod_type in model_types
@testset "$ex_name" for ex_name in eval(Symbol(mod_type, "_example_names"))

include(joinpath(examples_dir, ex_name, mod_type * ".jl"))
(ex_type, ex_insts) = include(joinpath(examples_dir, ex_name, mod_type * "_test.jl"))

for (inst_set, real_T, time_limit) in instance_sets
    haskey(ex_insts, inst_set) || continue
    inst_subset = ex_insts[inst_set]
    isempty(inst_subset) && continue

    info_perf = (; inst_set, real_T, :example => ex_name, :model_type => mod_type, :solver_options => ())
    new_default_options = (; default_options..., time_limit = time_limit)
    ex_type_T = ex_type{real_T}

    println("starting $ex_type_T $inst_set tests\n")
    @testset "$ex_type_T $inst_set" begin
        run_instance_set(inst_subset, ex_type_T, info_perf, new_default_options, script_verbose, perf, results_path)
    end
    println()
end

end
end

# println("\n")
# DataFrames.show(perf, allrows = true, allcols = true)
# println("\n")
end
;
