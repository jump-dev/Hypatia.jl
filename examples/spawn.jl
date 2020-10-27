#=
utilities for spawning benchmark runs
=#

# reduce printing for worker
Base.eval(Distributed, :(function redirect_worker_output(ident, stream)
    @async while !eof(stream)
        println(readline(stream))
    end
end))

function spawn_step(fun::Function, fun_name::Symbol, time_limit::Real, worker::Int)
    @assert nprocs() == 2
    status = :OK
    time_start = time()

    fut = Future()
    @async put!(fut, begin
        try
            remotecall_fetch(fun, worker)
        catch e
            println(fun_name, ": caught error: ", e)
        end
    end)

    killed_proc = false
    while !isready(fut)
        if Sys.free_memory() < free_memory_limit
            killstatus = :KilledMemory
        elseif time() - time_start > time_limit
            killstatus = :KilledTime
        else
            sleep(1)
            continue
        end
        try
            if !isready(fut)
                interrupt()
                sleep(5)
            end
            if !isready(fut) && nprocs() > 1
                w = workers()[end]
                run(`kill -SIGKILL $(remotecall_fetch(getpid, w))`)
                killed_proc = true
                sleep(10)
            end
        catch e
        end
        status = Symbol(fun_name, killstatus)
        println("status: ", status)
        break
    end

    output = (killed_proc || !isready(fut)) ? nothing : fetch(fut)
    if isnothing(output) && status == :OK
        status = Symbol(fun_name, :CaughtError)
    end
    finalize(fut)
    flush(stdout); flush(stderr)

    return (status, output)
end

function run_instance_check(
    ex_name::String,
    ex_type::Type{<:ExampleInstanceJuMP{Float64}},
    compile_inst::Tuple,
    inst::Tuple,
    extender,
    solver::Tuple,
    solve::Bool;
    )
    worker = addprocs(1, enable_threaded_blas = true, exeflags = `--threads $num_threads`)[1]
    @assert nprocs() == 2
    println("loading files")
    @fetchfrom worker begin
        @eval import LinearAlgebra
        LinearAlgebra.BLAS.set_num_threads(num_threads)
        @eval using MosekTools
        include(joinpath(examples_dir, "common_JuMP.jl"))
        include(joinpath(examples_dir, ex_name, "JuMP.jl"))
        # include(joinpath(examples_dir, ex_name, "JuMP_benchmark.jl"))
        flush(stdout); flush(stderr)
        return nothing
    end
    println("running compile instance")
    original_stdout = stdout
    (out_rd, out_wr) = redirect_stdout() # don't print output
    @fetchfrom worker begin
        run_instance(ex_type, compile_inst, extender, NamedTuple(), solver[2], default_options = solver[3], test = false)
        flush(stdout); flush(stderr)
        return nothing
    end
    redirect_stdout(original_stdout)
    close(out_wr)
    println("finished compile instance")

    println("\nsetup model")
    print_memory()
    setup_fun() = @eval begin
        (model, model_stats) = setup_model($ex_type, $inst, $extender, $(solver[3]), $(solver[2]))
        GC.gc()
        return model_stats
    end
    setup_time = @elapsed (status, model_stats) = spawn_step(setup_fun, :SetupModel, setup_time_limit, worker)
    setup_killed = (status != :OK)
    if setup_killed
        println("setup model failed: $status")
        model_stats = (-1, -1, -1, String[])
    end

    if solve && !setup_killed
        println("\nsolve and check")
        print_memory()
        check_fun() = @eval begin
            solve_stats = solve_check(model, test = false)
            return solve_stats
        end
        check_time = @elapsed (status, check_stats) = spawn_step(check_fun, :SolveCheck, check_time_limit, worker)
        check_killed = (status != :OK)
        check_killed && println("solve and check failed: $status")
    else
        check_time = 0.0
        check_killed = true
    end
    if check_killed
        if status == :OK
            @assert !solve
            status = :SkippedSolveCheck
        end
        check_stats = (string(status), NaN, -1, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
        solver_hit_limit = true
    else
        solver_status = string(check_stats[1])
        check_stats = (solver_status, check_stats[2:end]...)
        solver_hit_limit = (solver_status == "TimeLimit")
        solver_hit_limit && println("solver hit limit: $solver_status")
    end

    wait(rmprocs(worker))
    @assert nprocs() == 1

    return (setup_killed, solver_hit_limit, (model_stats..., check_stats..., setup_time, check_time))
end
