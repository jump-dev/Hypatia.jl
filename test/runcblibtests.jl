#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors
=#

examples_dir = joinpath(@__DIR__, "../examples")
include(joinpath(examples_dir, "common_JuMP.jl"))
using Test
using DataFrames
using Printf
using TimerOutputs
import Hypatia
import Hypatia.Solvers

# options to solvers
timer = TimerOutput()
tol = 1e-7
default_solver_options = (
    # verbose = false,
    verbose = true,
    iter_limit = 150,
    time_limit = 120,
    tol_rel_opt = tol,
    tol_abs_opt = tol,
    tol_feas = tol,
    # system_solver = Solvers.NaiveDenseSystemSolver{Float64}(),
    # system_solver = Solvers.SymIndefDenseSystemSolver{Float64}(),
    system_solver = Solvers.QRCholDenseSystemSolver{Float64}(),
    timer = timer,
    )

# instance sets to run and corresponding extenders and time limits (seconds)
include(joinpath(@__DIR__, "cblibsets.jl"))
instance_sets = [
    ("diverse_few", nothing, 15),
    ("power_small", nothing, 60),
    ("exp_small", nothing, 15),
    # ("exp_most", nothing, 60),
    ]

# CBLIB file folder location (use default)
cblib_dir = joinpath(ENV["HOME"], "cblib/cblib.zib.de/download/all")

# start the tests
@info("starting CBLIB tests")
for (inst_set, extender, time_limit) in instance_sets
    @info("each $inst_set instance with extender $extender should take <$time_limit seconds")
end

perf = DataFrame(
    inst_set = String[],
    count = Int[],
    inst = String[],
    test_time = Float64[],
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

all_tests_time = time()

@testset "CBLIB tests" begin
    for (inst_set, extender, time_limit) in instance_sets
        instances = eval(Symbol(inst_set))
        isempty(instances) && continue
        println("\nstarting $(length(instances)) instances for $inst_set with extender $extender\n")
        solver_options = (default_solver_options..., time_limit = time_limit)
        for (inst_num, inst) in enumerate(instances)
            test_info = "$inst_set $extender $inst_num: $inst"
            @testset "$test_info" begin
                println(test_info, "...")
                test_time = @elapsed r = test(inst, extender, solver_options)
                push!(perf, (
                    inst_set, inst_num, inst, test_time,
                    r.status, r.solve_time, r.num_iters, r.primal_obj, r.dual_obj,
                    r.obj_diff, r.compl, r.x_viol, r.y_viol, r.z_viol,
                    r.n, r.p, r.q,
                    ))
                @printf("... %8.2e seconds\n", test_time)
            end
        end
    end

    @printf("\ncblib tests total time: %8.2e seconds\n\n", time() - all_tests_time)
    @show sum(perf[:iters])
    show(perf, allrows = true, allcols = true)
    println("\n")
    # show(timer)
    # println("\n")
end

;
