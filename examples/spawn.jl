#=
utilities for spawning benchmark runs
=#

# reduce printing for worker
Base.eval(Distributed, :(function redirect_worker_output(ident, stream)
    @async while !eof(stream)
        println(readline(stream))
    end
end))

function kill_workers()
    if nprocs() > 1
        try
            interrupt()
        catch e
        end
        sleep(5)
    end
    if nprocs() > 1
        w = workers()[end]
        try
            run(`kill -SIGKILL $(remotecall_fetch(getpid, w))`)
        catch e
        end
        sleep(10)
    end
    @assert nprocs() == 1
end

function spawn_step(fun::Function, fun_name::Symbol, time_limit::Real)
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
    flush(stdout); flush(stderr)

    return (status, output)
end

function run_instance_check(
    ex_name::String,
    ex_type::Type{<:ExampleInstanceJuMP{Float64}},
    inst_data::Tuple,
    extender,
    solver::Tuple,
    solve::Bool,
    )
    if nprocs() < 2
        println("adding worker")
        addprocs(1, enable_threaded_blas = true, exeflags = `--threads $num_threads`)
        sleep(5)
        @assert nprocs() == 2
        global worker = workers()[end]
        @fetchfrom worker begin
            @eval import LinearAlgebra
            LinearAlgebra.BLAS.set_num_threads(num_threads)
            @eval using MosekTools
            include(joinpath(examples_dir, "common_JuMP.jl"))
            include(joinpath(examples_dir, ex_name, "JuMP.jl"))
            include(joinpath(examples_dir, ex_name, "JuMP_benchmark.jl"))
            flush(stdout); flush(stderr)
        end
        sleep(2)
    end

    println("setup model")
    setup_fun() = @eval begin
        (model, model_stats) = setup_model($ex_type, $inst_data, $extender, $(solver[3]), $(solver[2]))
        GC.gc()
        return model_stats
    end
    setup_time = @elapsed (status, model_stats) = spawn_step(setup_fun, :SetupModel, setup_time_limit)
    setup_killed = (status != :OK)
    if setup_killed
        println("setup model failed: $status")
        model_stats = (-1, -1, -1, String[])
    end

    if solve && !setup_killed
        println("solve and check")
        check_fun() = @eval begin
            solve_stats = solve_check(model, test = false)
            finalize(model)
            GC.gc()
            return solve_stats
        end
        check_time = @elapsed (status, check_stats) = spawn_step(check_fun, :SolveCheck, check_time_limit)
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

    return (setup_killed, solver_hit_limit, (model_stats..., check_stats..., setup_time, check_time))
end
