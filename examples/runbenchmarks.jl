#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

run benchmarks from the examples folder
to use the bench instance set and run on cmd line:
~/julia/julia examples/runbenchmarks.jl &> ~/bench/bench.txt
=#

import DataFrames
import CSV
using Printf
import TimerOutputs
import LinearAlgebra
using Hypatia
using MosekTools

examples_dir = @__DIR__
include(joinpath(examples_dir, "common_JuMP.jl"))

# path to write results DataFrame to CSV, if any
results_path = joinpath(homedir(), "bench", "bench.csv")
# results_path = nothing

# spawn_runs = true
spawn_runs = false

free_memory_limit = 16 * 2^30 # keep at least X GB of RAM available
optimizer_time_limit = 1800
solve_time_limit = 1.2 * optimizer_time_limit
setup_time_limit = optimizer_time_limit / 2

num_threads = Threads.nthreads()
blas_num_threads = LinearAlgebra.BLAS.get_num_threads()
@show num_threads
@show blas_num_threads
println()

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
    MSK_IPAR_NUM_THREADS = blas_num_threads,
    MSK_DPAR_OPTIMIZER_MAX_TIME = solve_time_limit,
    MSK_DPAR_INTPNT_CO_TOL_PFEAS = tol,
    MSK_DPAR_INTPNT_CO_TOL_DFEAS = tol,
    MSK_DPAR_INTPNT_CO_TOL_REL_GAP = tol,
    ))

# instance sets and solvers to run
instance_sets = [
    ("nat", [hyp_solver,]),
    ("ext", [
        hyp_solver,
        mosek_solver,
        ]),
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

function run_instance(
    ex_type::Type{<:ExampleInstanceJuMP},
    inst::Tuple,
    extender,
    solver::Tuple,
    )
    println("setup optimizer")
    setup_time = @elapsed (model, model_stats) = setup_model(ex_type{Float64}, inst, extender, solver[3], solver[2])

    println("solve and check")
    check_time = @elapsed solve_stats = solve_check(model, test = false)

    return (false, (model_stats..., solve_stats..., setup_time, check_time))
end

spawn_runs && include(joinpath(examples_dir, "spawn.jl"))

@info("starting script")

spawn_runs && spawn_setup()

perf = DataFrames.DataFrame(
    example = Type{<:ExampleInstance}[],
    inst_set = String[],
    count = Int[],
    inst_data = Tuple[],
    extender = Any[],
    solver = String[],
    n = Int[],
    p = Int[],
    q = Int[],
    cone_types = Vector{String}[],
    status = Symbol[],
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
    @everywhere (ex_type, ex_insts) = include(joinpath($examples_dir, $ex_name, "benchmark.jl"))

    for (inst_set, solvers) in instance_sets, solver in solvers
        haskey(ex_insts, inst_set) || continue
        (extender, inst_subsets) = ex_insts[inst_set]
        isempty(inst_subsets) && continue
        @info("starting instances for $ex_type $inst_set")

        for inst_subset in inst_subsets
            for (inst_num, inst) in enumerate(inst_subset)
                println("\n$ex_type $inst_set $(solver[1]) $inst_num: $inst ...")
                time_inst = @elapsed (is_killed, p) = run_instance(ex_type, inst, extender, solver)

                push!(perf, (ex_type, inst_set, inst_num, inst, extender, solver[1], p..., time_inst))
                isnothing(results_path) || CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
                @printf("... %8.2e seconds\n\n", time_inst)
                flush(stdout); flush(stderr)
                is_killed && break
            end
        end
    end
end

spawn_runs && kill_workers()

@printf("\nexamples tests total time: %8.2e seconds\n\n", time() - time_all)
DataFrames.show(perf, allrows = true, allcols = true)
println()
flush(stdout); flush(stderr)
;
