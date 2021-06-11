#=
run stepper benchmarks
see stepper/README.md
=#

using Test
using Printf
import DataFrames
import CSV
import Hypatia
import Hypatia.Solvers

include(joinpath(@__DIR__, "../../examples/Examples.jl"))
using Main.Examples

# path to write results DataFrame to CSV
results_path = joinpath(mkpath(joinpath(@__DIR__, "raw")), "bench.csv")

# script verbosity
script_verbose = false

# default options to solvers
default_options = (
    verbose = false,
    # verbose = true,
    default_tol_relax = 10,
    iter_limit = 10000,
    time_limit = 10000,
    )

# stepper option sets to run
porc = Solvers.PredOrCentStepper{Float64}
comb = Solvers.CombinedStepper{Float64}
stepper_options = [
    "basic" => porc(use_adjustment = false, use_curve_search = false,
        use_pred_sum_prox = true, use_sum_prox = true, prox_bound = 0.2844),
    "prox" => porc(use_adjustment = false, use_curve_search = false),
    "toa" => porc(use_adjustment = true, use_curve_search = false),
    "curve" => porc(use_adjustment = true, use_curve_search = true),
    "comb" => comb(shift_sched = 0),
    ]

# instance sets and real types to run and corresponding time limits (seconds)
inst_sets = [
    # "minimal",
    # "fast",
    "compile",
    "various",
    ]

time_all = time()

@testset "examples tests" begin
test_insts = Examples.get_test_instances()
perf = Examples.setup_benchmark_dataframe()
CSV.write(results_path, perf)

@testset "$mod, $ex" for (mod, mod_insts) in test_insts,
    (ex, (ex_type, ex_insts)) in mod_insts
@testset "$inst_set" for inst_set in inst_sets
    if inst_set == "compile"
        haskey(ex_insts, "various") || continue
        ex_insts["compile"] = ex_insts["various"]
    end
    haskey(ex_insts, inst_set) || continue
    inst_subset = ex_insts[inst_set]
    isempty(inst_subset) && continue

    for (step_name, stepper) in stepper_options
        info_perf = (; inst_set, :example => ex, :model_type => mod,
            :real_T => Float64, :solver_options => (step_name,))
        new_default_options = (; default_options..., stepper = stepper)

        str = "$mod $ex $inst_set $step_name"
        println("\nstarting $str")
        @testset "$str" begin
            Examples.run_instance_set(inst_subset, ex_type{Float64}, info_perf,
                new_default_options, script_verbose, perf, results_path)
        end
    end

end
end

println("\n")
DataFrames.show(perf, allrows = true, allcols = true)
println("\n")
flush(stdout); flush(stderr)
end

@printf("\nbenchmarks total time: %8.2e seconds\n\n", time() - time_all)
;
