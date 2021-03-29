include(joinpath(@__DIR__, "../setup.jl"))

# path to write results DataFrame to CSV, if any
results_path = joinpath(mkpath(joinpath(@__DIR__, "raw")), "bench.csv")
# results_path = nothing

# script verbosity
script_verbose = false

# default options to solvers
default_options = (
    verbose = false,
    # verbose = true,
    default_tol_relax = 10,
    iter_limit = 1000,
    time_limit = 3600,
    )

# stepper option sets to run
porc = Solvers.PredOrCentStepper{Float64}
comb = Solvers.CombinedStepper{Float64}
stepper_options = [
    "basic" => porc(use_correction = false, use_curve_search = false),
    "toa" => porc(use_correction = true, use_curve_search = false),
    "curve" => porc(use_correction = true, use_curve_search = true),
    "comb" => comb(),
    "shift" => comb(2),
    ]

# instance sets and real types to run and corresponding time limits (seconds)
instance_sets = [
    "minimal",
    "various",
    ]

# types of models to run and corresponding options and example names
model_types = [
    "native",
    "JuMP",
    ]

# list of names of native examples to run
native_example_names = [
    # "densityest",
    # "envelope",
    # "expdesign",
    "linearopt",
    "matrixcompletion",
    # "matrixregression",
    "maxvolume",
    # "polymin",
    # "portfolio",
    "sparsepca",
    ]

# list of names of JuMP examples to run
JuMP_example_names = [
    "centralpolymat",
    "conditionnum",
    "contraction",
    "densityest",
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
    "nearestpolymat",
    "regionofattr",
    "robustgeomprog",
    "semidefinitepoly",
    "shapeconregr",
    "signomialmin",
    "stabilitynumber",
    ]

perf = setup_benchmark_dataframe()
isnothing(results_path) || CSV.write(results_path, perf)
time_all = time()

@testset "examples tests" begin
@testset "$mod_type" for mod_type in model_types
@testset "$ex_name" for ex_name in eval(Symbol(mod_type, "_example_names"))

include(joinpath(examples_dir, ex_name, mod_type * ".jl"))
(ex_type, ex_insts) = include(joinpath(examples_dir, ex_name, mod_type * "_test.jl"))
ex_type_T = ex_type{Float64}

for inst_set in instance_sets, (step_name, stepper) in stepper_options
    haskey(ex_insts, inst_set) || continue
    inst_subset = ex_insts[inst_set]
    isempty(inst_subset) && continue

    info_perf = (; inst_set, :example => ex_name, :real_T => Float64, :solver_options => (step_name,))
    new_default_options = (; default_options..., stepper = stepper)

    println("\nstarting $ex_type $inst_set tests")
    @testset "$ex_type $inst_set" begin
        run_instance_set(inst_subset, ex_type_T, info_perf, new_default_options, script_verbose, perf, results_path)
    end
end

end
end

# println("\n")
# DataFrames.show(perf, allrows = true, allcols = true)
println("\n")
end

@printf("\nbenchmarks total time: %8.2e seconds\n\n", time() - time_all)
;
