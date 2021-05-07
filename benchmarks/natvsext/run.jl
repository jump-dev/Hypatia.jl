using Hypatia
using MosekTools
include(joinpath(@__DIR__, "../spawn.jl"))

# path to write results DataFrame to CSV, if any
results_path = joinpath(mkpath(joinpath(@__DIR__, "raw")), "bench.csv")
# results_path = nothing

setup_model_anyway = true # keep setting up models of larger size even if previous solve-check was killed
# setup_model_anyway = false

verbose = true # make solvers print output
# verbose = false
num_threads = 16 # number of threads to use for BLAS and Julia processes that run instances
free_memory_limit = 8 * 2^30 # keep at least X GB of RAM available
optimizer_time_limit = 1800
setup_time_limit = 2 * optimizer_time_limit
check_time_limit = 1.2 * optimizer_time_limit
tol_loose = 1e-7
tol_tight = 1e-3 * tol_loose

hyp_solver = ("Hypatia", Hypatia.Optimizer, (
    verbose = verbose,
    iter_limit = 250,
    time_limit = optimizer_time_limit,
    tol_abs_opt = tol_tight,
    tol_rel_opt = tol_loose,
    tol_feas = tol_loose,
    tol_infeas = tol_tight,
    init_use_indirect = true, # skips dual equalities preprocessing
    use_dense_model = true,
    ))
mosek_solver = ("Mosek", Mosek.Optimizer, (
    QUIET = !verbose,
    MSK_IPAR_NUM_THREADS = num_threads,
    MSK_IPAR_OPTIMIZER = Mosek.MSK_OPTIMIZER_CONIC,
    MSK_IPAR_INTPNT_BASIS = Mosek.MSK_BI_NEVER, # do not do basis identification for LO problems
    MSK_DPAR_OPTIMIZER_MAX_TIME = optimizer_time_limit,
    MSK_DPAR_INTPNT_CO_TOL_REL_GAP = tol_loose,
    MSK_DPAR_INTPNT_CO_TOL_PFEAS = tol_loose,
    MSK_DPAR_INTPNT_CO_TOL_DFEAS = tol_loose,
    MSK_DPAR_INTPNT_CO_TOL_INFEAS = tol_tight,
    ))

# instance sets and solvers to run
instance_sets = [
    ("nat", hyp_solver),
    ("ext", hyp_solver),
    ("ext", mosek_solver),
    ]

# models to run
JuMP_example_names = [
    # "densityest",
    # "doptimaldesign",
    # "matrixcompletion",
    # "matrixquadratic",
    # "matrixregression",
    # "nearestpsd",
    # "polymin",
    # "polynorm",
    # "portfolio",
    # "nearestpolymat",
    # "shapeconregr",
    ]

interrupt()
@assert nprocs() == 1
println()

for ex_name in JuMP_example_names
    include(joinpath(examples_dir, ex_name, "JuMP.jl"))
end

print_memory() = println("free memory (GB): ", Float64(Sys.free_memory()) / 2^30)
print_memory()

perf = setup_benchmark_dataframe()
isnothing(results_path) || CSV.write(results_path, perf)
time_all = time()

println("\nstarting benchmark runs\n")
for ex_name in JuMP_example_names
    (ex_type, ex_insts) = include(joinpath(examples_dir, ex_name, "JuMP_benchmark.jl"))
    ex_type_T = ex_type{Float64}

    for (inst_set, solver) in instance_sets
        haskey(ex_insts, inst_set) || continue
        (extender, inst_subsets) = ex_insts[inst_set]
        isempty(inst_subsets) && continue
        info_perf = (; inst_set, :extender => string(extender), :example => ex_name, :model_type => "JuMP", :real_T => Float64, :solver_options => (), :solver => solver[1])

        println("\nstarting instances for $ex_type $inst_set\n")
        for inst_subset in inst_subsets
            solve = true
            compile_inst = inst_subset[1] # first instance is only used for compilation
            for (inst_num, inst_data) in enumerate(inst_subset[2:end])
                println("\nstarting $ex_type $inst_set $(solver[1]) $inst_num: $inst_data ...\n")
                flush(stdout); flush(stderr)

                total_time = @elapsed (setup_killed, check_killed, run_perf) = spawn_instance(ex_name, ex_type_T, compile_inst, inst_data, extender, solver, solve)

                new_perf = (; info_perf..., run_perf..., total_time, inst_num, inst_data)
                write_perf(perf, results_path, new_perf)

                @printf("%8.2e seconds\n", total_time)
                flush(stdout); flush(stderr)

                setup_killed && break
                if check_killed
                    if setup_model_anyway
                        solve = false
                    else
                        break
                    end
                end
            end
        end
    end
end

interrupt()

# flush(stdout); flush(stderr)
# println("\n")
# DataFrames.show(perf, allrows = true, allcols = true)
println("\n")
flush(stdout); flush(stderr)

@printf("\nbenchmarks total time: %8.2e seconds\n\n", time() - time_all)
;
