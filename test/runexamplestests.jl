#=
run examples tests from the examples folder
=#

import Hypatia.Solvers
using Test
using Printf
import DataFrames
import CSV
examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "common_JuMP.jl"))
include(joinpath(examples_dir, "common_native.jl"))

# path to write results DataFrame to CSV, if any
# results_path = joinpath(homedir(), "bench", "bench.csv")
results_path = joinpath(pwd(), "bench2", "bench.csv")
# results_path = nothing

print_memory() = println("free memory (GB): ", Float64(Sys.free_memory()) / 2^30)

# default options to solvers
default_options = (
    # verbose = false,
    verbose = true,
    default_tol_relax = 10,
    iter_limit = 1000,
    )

predorcent = Solvers.PredOrCentStepper{Float64}
combined = Solvers.CombinedStepper{Float64}
qrchol = Solvers.QRCholDenseSystemSolver{Float64}()

stepper_solvers = [
    (predorcent(use_correction = false, use_curve_search = false), [qrchol]),
    (predorcent(use_correction = true, use_curve_search = false), [qrchol]),
    (predorcent(use_correction = true, use_curve_search = true), [qrchol]),
    (combined(), [qrchol]),
    (combined(2), [qrchol]),
    ]

# instance sets and real types to run and corresponding time limits (seconds)
instance_sets = [
    ("minimal", Float64, 100),
    # ("minimal", Float32, 60),
    # ("minimal", BigFloat, 60),
    # ("fast", Float64, 300),
    # ("slow", Float64, 120),
    # ("compile", Float64, 3600),
    ("various", Float64, 3600),
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
    # "matrixcompletion",
    # "matrixregression",
    # "maxvolume",
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
    "polymin",
    "polynorm",
    "portfolio",
    "nearestpolymat",
    "regionofattr",
    "robustgeomprog",
    "normconepoly",
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
    stepper = Symbol[],
    system_solver = Symbol[],
    toa = Bool[],
    curve = Bool[],
    shift = Int[],
    n = Int[],
    p = Int[],
    q = Int[],
    nu = Float64[],
    cone_types = Vector{String}[],
    num_cones = Int[],
    maxq = Int[],
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
    time_rescale = Float64[],
    time_initx = Float64[],
    time_inity = Float64[],
    time_unproc = Float64[],
    time_loadsys = Float64[],
    time_upsys = Float64[],
    time_upfact = Float64[],
    time_uprhs = Float64[],
    time_getdir = Float64[],
    time_search = Float64[],
    setup_time = Float64[],
    check_time = Float64[],
    total_time = Float64[],
    )

isnothing(results_path) || CSV.write(results_path, perf)

# TODO decide what to do with this.
# counts number of instances with loosened tols
# n = 1
# exs = []
# for ex_name in eval(Symbol("JuMP", "_example_names"))
#     include(joinpath(examples_dir, ex_name, "JuMP" * ".jl"))
#     (ex_type, ex_insts) = include(joinpath(examples_dir, ex_name, "JuMP" * "_test.jl"))
#     inst_subset = ex_insts["various"]
#     for (inst_num, inst) in enumerate(inst_subset)
#         if length(inst) == 3
#             global n += 1
#             push!(exs, ex_name)
#             @show ex_name
#             @show inst[3].default_tol_relax
#         end
#     end
# end
# @show n
# @show unique(exs)

@testset "examples tests" begin
@testset "$mod_type" for mod_type in model_types
@testset "$ex_name" for ex_name in eval(Symbol(mod_type, "_example_names"))
include(joinpath(examples_dir, ex_name, mod_type * ".jl"))
(ex_type, ex_insts) = include(joinpath(examples_dir, ex_name, mod_type * "_test.jl"))

for (stepper, system_solvers) in stepper_solvers, system_solver in system_solvers, (inst_set, real_T, time_limit) in instance_sets
    print_memory()
    (inst_set == "compile") && (ex_insts["compile"] = ex_insts["various"])
    haskey(ex_insts, inst_set) || continue
    inst_subset = ex_insts[inst_set]
    isempty(inst_subset) && continue
    ex_type_T = ex_type{real_T}
    new_default_options = (; default_options..., time_limit = time_limit, stepper = stepper, system_solver = system_solver)

    println("\nstarting $ex_type_T $inst_set tests")
    @testset "$ex_type_T $inst_set" begin
    for (inst_num, inst) in enumerate(inst_subset)
        test_info = "inst $inst_num: $(inst[1])"
        @testset "$test_info" begin
            println(test_info, " ...")
            test_time = @elapsed p = run_instance(ex_type_T, inst..., default_options = new_default_options, verbose = false)
            extender = (length(inst) > 1 && mod_type == "JuMP") ? inst[2] : nothing
            push!(perf, (string(ex_type), inst_set, real_T, inst_num, inst[1], string(extender), nameof(typeof(stepper)), nameof(typeof(system_solver)),
                use_correction(stepper), use_curve_search(stepper), shift(stepper), p..., test_time))
            isnothing(results_path) || CSV.write(results_path, perf[end:end, :], transform = (col, val) -> something(val, missing), append = true)
            @printf("%8.2e seconds\n", test_time)
        end
    end
    end
end
end
end

# println("\n")
# DataFrames.show(perf, allrows = true, allcols = true)
# println("\n")
# @show sum(perf[!, :iters])
# @show sum(perf[!, :solve_time])
end
;
