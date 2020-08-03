#=
Copyright 2019, Chris Coey and contributors

run benchmarks from the examples folder
to use the bench instance set and run on cmd line:
~/julia/julia test/runexamplestests.jl &> ~/bench/bench.txt
=#

examples_dir = @__DIR__
include(joinpath(examples_dir, "common.jl"))
include(joinpath(examples_dir, "common_JuMP.jl"))
# include(joinpath(examples_dir, "common_native.jl"))
import DataFrames
import CSV
using Printf
import TimerOutputs
import DataStructures
# import Mosek

# path to write results DataFrame to CSV, if any
results_path = joinpath(homedir(), "bench", "bench.csv")
# results_path = nothing

# options to solvers
hyp_solver = ("Hypatia", Hypatia.Optimizer, (
    verbose = true,
    iter_limit = 250,
    ))
# mosek_solver = ("Mosek", Mosek.Optimizer, (
#     verbose = true,
#     ))

# instance sets and solvers to run
instance_sets = [
    ("nat", [hyp_solver,]),
    ("ext", [
        hyp_solver,
        # mosek_solver,
        ]),
    ]

# models to run
model_types = [
    # "native",
    "JuMP",
    ]
native_example_names = [
    ]
JuMP_example_names = [
    "densityest",
    "expdesign",
    "matrixcompletion",
    "matrixquadratic",
    "matrixregression",
    "nearestpsd",
    "polymin",
    "portfolio",
    "shapeconregr",
    ]

# start the benchmarks
@info("starting benchmark runs")

# load instances
# TODO don't load unwanted instance sets
instances = DataStructures.DefaultOrderedDict{Type{<:ExampleInstance}, Any}(() -> DataStructures.DefaultOrderedDict{String, Any}(() -> []))
for mod_type in model_types, ex_name in eval(Symbol(mod_type, "_example_names"))
    include(joinpath(examples_dir, ex_name, mod_type * ".jl"))
    include(joinpath(examples_dir, ex_name, "benchmark.jl"))
end

perf = DataFrames.DataFrame(
    example = Type{<:ExampleInstance}[],
    inst_set = String[],
    count = Int[],
    inst_data = Tuple[],
    solver = String[],
    extender = Any[],
    test_time = Float64[],
    build_time = Float64[],
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
    ext_n = Int[],
    ext_p = Int[],
    ext_q = Int[],
    )

isnothing(results_path) || CSV.write(results_path, perf)
all_tests_time = time()

@testset "examples tests" begin
    for (ex_type, ex_insts) in instances, (inst_set, solvers) in instance_sets, solver in solvers
        (extender, inst_subsets) = ex_insts[inst_set]
        isempty(inst_subsets) && continue
        println("\nstarting instances for $ex_type $inst_set\n")
        for inst_subset in inst_subsets
            # TODO spawn here
            for (inst_num, inst) in enumerate(inst_subset)
                test_info = "$ex_type $inst_set $(solver[1]) $inst_num: $inst"
                @testset "$test_info" begin
                    println(test_info, "...")
                    test_time = @elapsed (_, build_time, r) = test(ex_type{Float64}, inst, extender, solver[3], solver[2])
                    push!(perf, (
                        ex_type, inst_set, inst_num, inst, solver[1], extender, test_time, build_time,
                        r.status, r.solve_time, r.num_iters, r.primal_obj, r.dual_obj,
                        r.obj_diff, r.compl, r.x_viol, r.y_viol, r.z_viol,
                        r.n, r.p, r.q,
                        ))
                    isnothing(results_path) || CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append=true)
                    @printf("... %8.2e seconds\n", test_time)
                    flush(stdout)
                    flush(stderr)
                end
            end
        end
    end

    @printf("\nexamples tests total time: %8.2e seconds\n\n", time() - all_tests_time)
    DataFrames.show(perf, allrows = true, allcols = true)
    println("\n")
    @show sum(perf[:iters])
end

;
