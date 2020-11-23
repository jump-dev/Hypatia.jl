#=
run benchmarks from the examples folder
to use the bench instance set and run on cmd line:
killall julia; ~/julia/julia examples/runbenchmarks.jl &> ~/bench/bench.txt
=#

import DataFrames
import CSV
using Printf
import LinearAlgebra
using Distributed
using Hypatia
using MosekTools

interrupt()
@assert nprocs() == 1
println()

examples_dir = @__DIR__
include(joinpath(examples_dir, "common_JuMP.jl"))

# path to write results DataFrame to CSV, if any
results_path = joinpath(homedir(), "bench", "bench.csv")
# results_path = nothing

# spawn_runs = true # spawn new process for each instance
spawn_runs = false

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
    # ("ext", hyp_solver),
    ("ext", mosek_solver),
    ]

# models to run
JuMP_example_names = [
    # "densityest",
    # "expdesign",
    # "matrixcompletion",
    # "matrixquadratic",
    "matrixregression",
    # "nearestpsd",
    # "polymin",
    # "polynorm",
    # "portfolio",
    # "randompolymat",
    # "shapeconregr",
    ]

for ex_name in JuMP_example_names
    include(joinpath(examples_dir, ex_name, "JuMP.jl"))
end

print_memory() = println("free memory (GB): ", Float64(Sys.free_memory()) / 2^30)

print_memory()

if spawn_runs
    include(joinpath(examples_dir, "spawn.jl"))
else
    LinearAlgebra.BLAS.set_num_threads(num_threads)
    function run_instance_check(
        ::String,
        ex_type::Type{<:ExampleInstanceJuMP{Float64}},
        ::Tuple,
        inst_data::Tuple,
        extender,
        solver::Tuple,
        ::Bool,
        )
        return (false, false, run_instance(ex_type, inst_data, extender, NamedTuple(), solver[2], default_options = solver[3], test = false))
    end
end

perf = DataFrames.DataFrame(
    example = String[],
    inst_set = String[],
    count = Int[],
    inst_data = Tuple[],
    extender = String[],
    solver = String[],
    n = Int[],
    p = Int[],
    q = Int[],
    nu = Float64[],
    cone_types = Vector{String}[],
    status = String[],
    solve_time = Float64[],
    iters = Int[],
    prim_obj = Float64[],
    dual_obj = Float64[],
    rel_obj_diff = Float64[],
    compl = Float64[],
    x_viol = Float64[],
    y_viol = Float64[],
    z_viol = Float64[],
    setup_time = Float64[],
    check_time = Float64[],
    total_time = Float64[],
    )
DataFrames.allowmissing!(perf, 7:21)

isnothing(results_path) || CSV.write(results_path, perf)
time_all = time()

@info("starting benchmark runs")
for ex_name in JuMP_example_names
    (ex_type, ex_insts) = include(joinpath(examples_dir, ex_name, "JuMP_benchmark.jl"))

    for (inst_set, solver) in instance_sets
        haskey(ex_insts, inst_set) || continue
        (extender, inst_subsets) = ex_insts[inst_set]
        isempty(inst_subsets) && continue
        @info("starting instances for $ex_type $inst_set")

        for inst_subset in inst_subsets
            solve = true
            compile_inst = inst_subset[1]
            for (inst_num, inst) in enumerate(inst_subset[2:end])
                println()
                @info("starting $ex_type $inst_set $(solver[1]) $inst_num: $inst ...")
                flush(stdout); flush(stderr)

                time_inst = @elapsed (setup_killed, check_killed, p) = run_instance_check(ex_name, ex_type{Float64}, compile_inst, inst, extender, solver, solve)

                push!(perf, (string(ex_type), inst_set, inst_num, inst, string(extender), solver[1], p..., time_inst))
                isnothing(results_path) || CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
                @printf("... %8.2e seconds\n\n", time_inst)
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

@printf("\nbenchmarks total time: %8.2e seconds\n\n", time() - time_all)
DataFrames.show(perf, allrows = true, allcols = true)
println()
spawn_runs && interrupt()
flush(stdout); flush(stderr)
;
