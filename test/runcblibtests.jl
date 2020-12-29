#=
run CBLIB tests
=#

using Test
using Printf
import DataFrames
import CSV
include(joinpath(@__DIR__, "cblibsets.jl"))
examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "common_JuMP.jl"))

# CBLIB file folder location (use default)
cblib_dir = joinpath(ENV["HOME"], "cblib/cblib.zib.de/download/all")

# path to write results DataFrame to CSV, if any
# results_path = joinpath(homedir(), "bench", "bench.csv")
results_path = nothing

# instance sets to run and corresponding time limits (seconds)
instance_sets = [
    ("diverse_few", 15),
    # ("power_small", 60),
    # ("exp_small", 15),
    # ("exp_most", 60),
    ]

# options to solvers
default_options = (
    verbose = false,
    # verbose = true,
    iter_limit = 150,
    time_limit = 120,
    default_tol_relax = 100,
    # system_solver = Solvers.NaiveDenseSystemSolver{Float64}(),
    system_solver = Solvers.SymIndefSparseSystemSolver{Float64}(),
    # system_solver = Solvers.QRCholDenseSystemSolver{Float64}(),
    )

perf = DataFrames.DataFrame(
    inst_set = String[],
    count = Int[],
    inst = String[],
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

isnothing(results_path) || CSV.write(results_path, perf)

@testset "CBLIB tests" begin
for (inst_set, time_limit) in instance_sets
    inst_subset = eval(Symbol(inst_set))
    isempty(inst_subset) && continue
    solver_options = (; default_options..., time_limit = time_limit)
    println("\nstarting $(length(inst_subset)) instances for $inst_set\n")

    for (inst_num, inst) in enumerate(inst_subset)
        test_info = "$inst_set $inst_num: $inst"
        @testset "$test_info" begin
            println(test_info, "...")
            time_inst = @elapsed p = run_cbf(inst, solver_options)
            push!(perf, (inst_set, inst_num, inst, p..., time_inst))
            isnothing(results_path) || CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
            @printf("... %8.2e seconds\n\n", time_inst)
        end
    end
end

DataFrames.show(perf, allrows = true, allcols = true)
println("\n")
end
;
