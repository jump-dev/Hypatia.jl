#=
Copyright 2020, Chris Coey, Lea Kapelevich and contributors

utilities for spawning benchmark runs
=#

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
    @fetchfrom worker @eval using MosekTools
    @fetchfrom worker include(joinpath(examples_dir, "common_JuMP.jl"))
end

function spawn_step(fun::Function, fun_name::Symbol)
    @assert nprocs() == 2
    status = :ok
    output = nothing
    time_start = time()

    fut = Future()
    @async put!(fut, begin
        try
            remotecall_fetch(fun, worker)
        catch e
            println(fun_name, ": caught error: ", e)
        end
    end)

    while !isready(fut)
        if Sys.free_memory() < free_memory_limit
            killstatus = :KilledMemory
        elseif time() - time_start > setup_time_limit
            killstatus = :KilledTime
        else
            sleep(1)
            continue
        end
        interrupt()
        sleep(1)
        isready(fut) || kill_workers()
        sleep(1)
        status = Symbol(fun_name, killstatus)
        println("status: ", status)
        break
    end

    output = isready(fut) ? fetch(fut) : nothing
    if isnothing(output) && status == :ok
        status = Symbol(fun_name, :CaughtError)
    end
    flush(stdout); flush(stderr)

    if nprocs() < 2
        get_worker()
        @fetchfrom worker begin
            Base.@eval using MosekTools
            include(joinpath(examples_dir, "common_JuMP.jl"))
            include(joinpath(examples_dir, ex_name, "JuMP.jl"))
            include(joinpath(examples_dir, ex_name, "benchmark.jl"))
            flush(stdout); flush(stderr)
        end
    end
    return (status, output)
end

function spawn_instance_check(
    ex_type::Type{<:ExampleInstanceJuMP{Float64}},
    inst_data::Tuple,
    extender,
    solver::Tuple,
    )
    println("setup model")
    fun = () -> setup_model(ex_type, inst_data, extender, solver[3], solver[2])
    setup_time = @elapsed (status, output) = spawn_step(fun, :SetupModel)

    check_time = @elapsed if status == :ok
        (model, model_stats) = output
        println("solve and check")
        fun = () -> solve_check(model, test = false)
        (status, solve_stats) = spawn_step(fun, :SolveCheck)
    else
        model_stats = (-1, -1, -1, String[])
    end

    if status != :ok
        solve_stats = (status, NaN, -1, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    end

    return (status != :ok, (model_stats..., solve_stats..., setup_time, check_time))
end
