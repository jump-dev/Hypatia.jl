#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

utilities for spawning benchmark runs
=#

using Distributed

# reduce printing for worker
Base.eval(Distributed, :(function redirect_worker_output(ident, stream)
    @async while !eof(stream)
        println(readline(stream))
    end
end))

function get_worker()
    if nprocs() < 2
        println("adding worker")
        addprocs(1, enable_threaded_blas = true, exeflags = `--threads $num_threads`)
        sleep(1)
    end
    global worker = workers()[end]
    return nothing
end

function kill_workers()
    for w in workers()[2:end]
        println("killing worker $w")
        @spawnat w begin flush(stdout); flush(stderr) end
        sleep(1)
        run(`kill -SIGKILL $(remotecall_fetch(getpid, w))`)
    end
    sleep(1)
end

function spawn_setup()
    kill_workers()
    get_worker()
    @spawnat worker @eval using MosekTools
    @spawnat worker include(joinpath(examples_dir, "common_JuMP.jl"))
end

function spawn_step(fun)
    @assert nprocs() == 2
    status = :ok
    output = nothing
    time_start = time()

    fut = Future()
    @async put!(fut, begin
        try
            remotecall_fetch(fun, worker)
        catch e
            println("caught error: ", e)
        end
    end)

    while !isready(fut)
        if Sys.free_memory() < free_memory_limit
            status = :KilledMemory
            println("killed memory")
        elseif time() - time_start > setup_time_limit
            status = :KilledTime
            println("killed time")
        else
            sleep(1)
            continue
        end
        interrupt()
        sleep(1)
        isready(fut) || kill_workers()
        sleep(1)
        break
    end

    output = isready(fut) ? fetch(fut) : nothing
    if isnothing(output) && status == :ok
        status = :CaughtError
    end
    flush(stdout); flush(stderr)

    if nprocs() < 2
        get_worker()
        @spawnat worker begin
            Base.@eval using MosekTools
            include(joinpath(examples_dir, "common_JuMP.jl"))
            include(joinpath(examples_dir, ex_name, "JuMP.jl"))
            include(joinpath(examples_dir, ex_name, "benchmark.jl"))
            flush(stdout); flush(stderr)
        end
    end
    return (status, output)
end

function run_instance(
    ex_type::Type{<:ExampleInstanceJuMP},
    inst::Tuple,
    extender,
    solver::Tuple,
    )
    println("setup model")
    fun = () -> setup_model(ex_type{Float64}, inst, extender, solver[3], solver[2])
    setup_time = @elapsed (status, output) = spawn_step(fun)

    check_time = @elapsed if status == :ok
        (model, model_stats) = output
        println("solve and check")
        fun = () -> solve_check(model, test = false)
        (status, solve_stats) = spawn_step(fun)
    else
        model_stats = (-1, -1, -1, String[])
    end

    if status != :ok
        solve_stats = (status, NaN, -1, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    end

    return (status != :ok, (model_stats..., solve_stats..., setup_time, check_time))
end
