#=
run benchmarks from the examples folder
to use the bench instance set and run on cmd line:
~/julia/julia examples/runbenchmarks.jl &> ~/bench/bench.txt
=#

import DataFrames
import CSV
using Printf
import LinearAlgebra
using Distributed
using Hypatia
using MosekTools

# num_threads = 16 # number of threads to use for BLAS and Julia processes that run instances
num_threads = 8 # number of threads to use for BLAS and Julia processes that run instances
LinearAlgebra.BLAS.set_num_threads(num_threads)
println()

examples_dir = @__DIR__
include(joinpath(examples_dir, "common_JuMP.jl"))

# path to write results DataFrame to CSV, if any
results_path = joinpath(homedir(), "bench", "bench.csv")
# results_path = nothing

# spawn_runs = true # needed for running Julia process with multiple threads
spawn_runs = false

free_memory_limit = 16 * 2^30 # keep at least X GB of RAM available
# optimizer_time_limit = 1800
optimizer_time_limit = 60
solve_time_limit = 1.2 * optimizer_time_limit
setup_time_limit = optimizer_time_limit

# options to solvers
tol = 1e-7
hyp_solver = ("Hypatia", Hypatia.Optimizer, (
    verbose = true,
    iter_limit = 250,
    time_limit = solve_time_limit,
    tol_abs_opt = tol,
    tol_rel_opt = tol,
    tol_feas = tol,
    ))
mosek_solver = ("Mosek", Mosek.Optimizer, (
    QUIET = false,
    MSK_IPAR_NUM_THREADS = num_threads,
    MSK_IPAR_OPTIMIZER = Mosek.MSK_OPTIMIZER_CONIC,
    MSK_IPAR_INTPNT_BASIS = Mosek.MSK_BI_NEVER, # do not do basis identification for LO problems
    MSK_DPAR_OPTIMIZER_MAX_TIME = solve_time_limit,
    MSK_DPAR_INTPNT_CO_TOL_PFEAS = tol,
    MSK_DPAR_INTPNT_CO_TOL_DFEAS = tol,
    MSK_DPAR_INTPNT_CO_TOL_REL_GAP = tol,
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
    # "expdesign",
    # "matrixcompletion",
    # "matrixquadratic",
    # "matrixregression",
    # "nearestpsd",
    # "polymin",
    # "portfolio",
    # "shapeconregr",
    ]

function run_instance_check(
    ex_type::Type{<:ExampleInstanceJuMP{Float64}},
    inst_data::Tuple,
    extender,
    solver::Tuple,
    )
    return (false, run_instance(ex_type, inst_data, extender, NamedTuple(), solver[2], default_options = solver[3], test = false))
end

if spawn_runs
    include(joinpath(examples_dir, "spawn.jl"))
    spawn_setup()
    instance_check_fun = spawn_instance_check
else
    instance_check_fun = run_instance_check
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
    cone_types = Vector{String}[],
    status = String[],
    solve_time = Float64[],
    iters = Int[],
    prim_obj = Float64[],
    dual_obj = Float64[],
    obj_diff = Float64[],
    compl = Float64[],
    x_viol = Float64[],
    y_viol = Float64[],
    z_viol = Float64[],
    setup_time = Float64[],
    check_time = Float64[],
    total_time = Float64[],
    )

isnothing(results_path) || CSV.write(results_path, perf)
time_all = time()

@info("starting benchmark runs")
for ex_name in JuMP_example_names
    @everywhere include(joinpath($examples_dir, $ex_name, "JuMP.jl"))
    @everywhere (ex_type, ex_insts) = include(joinpath($examples_dir, $ex_name, "JuMP_benchmark.jl"))

    for (inst_set, solver) in instance_sets
        haskey(ex_insts, inst_set) || continue
        (extender, inst_subsets) = ex_insts[inst_set]
        isempty(inst_subsets) && continue
        @info("starting instances for $ex_type $inst_set")

        for inst_subset in inst_subsets
            for (inst_num, inst) in enumerate(inst_subset)
                println("\n$ex_type $inst_set $(solver[1]) $inst_num: $inst ...")
                time_inst = @elapsed (is_killed, p) = instance_check_fun(ex_type{Float64}, inst, extender, solver)

                push!(perf, (string(ex_type), inst_set, inst_num, inst, string(extender), solver[1], p..., time_inst))
                isnothing(results_path) || CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
                @printf("... %8.2e seconds\n\n", time_inst)
                flush(stdout); flush(stderr)

                if is_killed
                    spawn_runs && spawn_reload(ex_name)
                    break
                end
            end
        end
    end
    sleep(1)
end

spawn_runs && kill_workers()

@printf("\nbenchmarks total time: %8.2e seconds\n\n", time() - time_all)
DataFrames.show(perf, allrows = true, allcols = true)
println()
flush(stdout); flush(stderr)
;
