#=
run examples tests from the examples folder
=#

using Test
using Printf
import DataFrames
import CSV
examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "common_JuMP.jl"))
include(joinpath(examples_dir, "common_native.jl"))

# path to write results DataFrame to CSV, if any
# results_path = joinpath(homedir(), "bench", "bench.csv")
results_path = nothing

# options to solvers
# tol = 1e-10
default_options = (
    verbose = false,
    # verbose = true,
    # tol_abs_opt = tol,
    # tol_rel_opt = tol,
    # tol_feas = tol,
    )

# instance sets and real types to run and corresponding time limits (seconds)
instance_sets = [
    ("minimal", Float64, 30),
    # ("minimal", Float32, 30),
    # ("minimal", BigFloat, 30),
    # ("fast", Float64, 30),
    # ("slow", Float64, 120),
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
    "polymin",
    "polynorm",
    "portfolio",
    "regionofattr",
    "robustgeomprog",
    "secondorderpoly",
    "semidefinitepoly",
    "shapeconregr",
    "signomialmin",
    "stabilitynumber",
    ]

perf = DataFrames.DataFrame(
    example = String[],
    inst_set = String[],
    real_T = Type{<:Real}[],
    count = Int[],
    inst_data = Tuple[],
    extender = String[],
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

@testset "examples tests" begin
@testset "$ex_name" for mod_type in model_types, ex_name in eval(Symbol(mod_type, "_example_names"))
include(joinpath(examples_dir, ex_name, mod_type * ".jl"))
(ex_type, ex_insts) = include(joinpath(examples_dir, ex_name, mod_type * "_test.jl"))

for (inst_set, real_T, time_limit) in instance_sets
    haskey(ex_insts, inst_set) || continue
    inst_subset = ex_insts[inst_set]
    isempty(inst_subset) && continue
    ex_type_T = ex_type{real_T}
    new_default_options = (; default_options..., time_limit = time_limit)

    println("\nstarting $ex_type_T $inst_set tests")
    @testset "$ex_type_T $inst_set" begin
    for (inst_num, inst) in enumerate(inst_subset)
        test_info = "inst $inst_num: $(inst[1])"
        @testset "$test_info" begin
            println(test_info, " ...")
            test_time = @elapsed p = run_instance(ex_type_T, inst..., default_options = new_default_options, verbose = false)
            extender = (length(inst) > 1 && mod_type == "JuMP") ? inst[2] : nothing
            push!(perf, (string(ex_type), inst_set, real_T, inst_num, inst[1], string(extender), p..., test_time))
            isnothing(results_path) || CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
            @printf("%8.2e seconds\n", test_time)
        end
    end
    end
end
end

# @printf("\nexamples tests total time: %8.2e seconds\n\n", time() - time_all)
# DataFrames.show(perf, allrows = true, allcols = true)
# println("\n")
end
;
