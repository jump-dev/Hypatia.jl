#=
run examples tests from the examples folder
=#

using Test
import DataFrames
include(joinpath(@__DIR__, "../examples/Examples.jl"))
using Main.Examples

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
inst_sets = [
    ("minimal", Float64, 60),
    # ("minimal", Float32, 60),
    # ("minimal", BigFloat, 60),
    # ("fast", Float64, 60),
    # ("various", Float64, 120),
    ]

perf = Examples.setup_benchmark_dataframe()

@testset "examples tests" begin
test_insts = Examples.get_test_instances()

@testset "$mod, $ex" for (mod, mod_insts) in test_insts,
    (ex, (ex_type, ex_insts)) in mod_insts
@testset "$inst_set, $T, $time_limit" for (inst_set, T, time_limit) in inst_sets
    haskey(ex_insts, inst_set) || continue
    inst_subset = ex_insts[inst_set]
    isempty(inst_subset) && continue

    info_perf = (; inst_set, :real_T => T, :example => ex, :model_type => mod)
    new_default_options = (; default_options..., time_limit = time_limit)
    ex_type_T = ex_type{T}

    str = "$mod $ex $T $inst_set"
    println("\nstarting $str tests")
    @testset "$str" begin
        Examples.run_instance_set(inst_subset, ex_type_T, info_perf,
            new_default_options, script_verbose, perf)
    end
end
end

# println("\n")
# DataFrames.show(perf, allrows = true, allcols = true)
# println("\n")
end
;
