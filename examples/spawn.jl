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
    try
        interrupt()
    catch e
    end
    sleep(5)
    while nprocs() > 1
        w = workers()[end]
        try
            run(`kill -SIGKILL $(remotecall_fetch(getpid, w))`)
        catch e
        end
        sleep(5)
    end
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
        try
            isready(fut) || interrupt()
        catch e
        end
        sleep(5)
        isready(fut) || kill_workers()
        status = Symbol(fun_name, killstatus)
        println("status: ", status)
        break
    end

    output = isready(fut) ? fetch(fut) : nothing
    if isnothing(output) && status == :ok
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
        return model_stats
    end
    setup_time = @elapsed (setup_status, model_stats) = spawn_step(setup_fun, :SetupModel)

    if setup_status == :ok
        println("solve and check")
        solve_fun() = @eval solve_check(model, test = false)
        check_time = @elapsed (status, solve_stats) = spawn_step(solve_fun, :SolveCheck)
    else
        check_time = 0.0
        model_stats = (-1, -1, -1, String[])
        status = setup_status
    end

    if status != :ok
        solve_stats = (status, NaN, -1, NaN, NaN, NaN, NaN, NaN, NaN, NaN)
    end

    return (setup_status != :ok, (model_stats..., string(solve_stats[1]), solve_stats[2:end]..., setup_time, check_time))
end
