using Printf
using CSV
using DataFrames

bench_file = joinpath(@__DIR__, "raw", "bench.csv")
output_folder = mkpath(joinpath(@__DIR__, "analysis"))

# uncomment examples to process
examples_params = Dict(
    # Hypatia paper examples:
    "densityest" => (
        [:m, :twok], [2, 3],
        [:SEP,], [:nu_nat, :n_nat, :n_SEP]
        ),
    "expdesign" => (
        [:logdet, :k], [5, 1],
        # [:EP,], [:n_nat, :n_EP, :q_nat, :q_EP]
        # [:SEP,], [:n_SEP, :q_nat, :q_SEP]
        [:EP, :SEP], [:nu_nat, :n_nat, :q_nat, :nu_EP, :n_EP, :q_EP, :nu_SEP, :n_SEP, :q_SEP]
        ),
    "matrixcompletion" => (
        [:m, :k], [1, 2],
        # [:EP,], [:n_EP, :p_nat, :q_EP]
        # [:SEP,], [:n_SEP, :p_nat, :q_SEP]
        [:EP, :SEP], [:nu_nat, :n_nat, :p_nat, :q_nat, :nu_EP, :n_EP, :q_EP, :nu_SEP, :n_SEP, :q_SEP]
        ),
    "matrixregression" => (
        [:m, :k], [2, 1],
        [:SEP,], [:n_SEP, :q_nat]
        ),
    "nearestpsd" => (
        [:compl, :k], [2, 1],
        [:SEP,], [:n_nat, :q_SEP]
        ),
    "polymin" => (
        [:m, :k], [1, 2],
        [:SEP,], [:nu_nat, :n_nat, :q_SEP]
        ),
    "portfolio" => (
        [:k], [1],
        [:SEP,], Symbol[]
        ),
    "shapeconregr" => (
        [:m, :twok], [1, 5],
        [:SEP,], [:nu_nat, :n_nat, :n_SEP, :q_nat]
        ),
    # SOS paper examples:
    "polynorm" => (
        [:L1, :n, :d, :m], [5, 1, 3, 4],
        [:SEP,], Symbol[]
        ),
    "nearestpolymat" => (
        [:n, :d, :m], [1, 2, 3],
        [:SEP,], Symbol[]
        ),
    )

print_table_solvers =
    true # add solver results to tex tables
    # false # just put formulation sizes in tex tables

@info("running examples: $(keys(examples_params))")

function post_process()
    all_df = CSV.read(bench_file, DataFrame)
    replace!(all_df.extender, extender_map...)
    for (ex_name, ex_params) in examples_params
        println()
        ex_df = filter(t -> t.example == ex_name && t.extender in vcat("", string.(ex_params[3])), all_df)
        if isempty(ex_df)
            @info("no data for $ex_name with params: $ex_params")
            continue
        end
        @info("starting $ex_name with params: $ex_params")
        (ex_df_wide, inst_solvers) = make_wide_csv(ex_df, ex_name, ex_params)
        make_table_tex(ex_name, ex_params, ex_df_wide, inst_solvers)
        make_plot_csv(ex_name, ex_params, ex_df_wide, inst_solvers)
        @info("finished $ex_name")
    end
    println()
    @info("finished all")
end

extender_map = Dict(
    "nothing" => "",
    "ExpPSDOptimizer" => "EP",
    "SOCExpPSDOptimizer" => "SEP",
    )

status_map = Dict(
    "SetupModelCaughtError" => "m",
    "SetupModelKilledTime" => "m",
    "SetupModelKilledMemory" => "m",
    "Optimal" => "co",
    "TimeLimit" => "tl",
    "SolveCheckKilledTime" => "tl",
    "SolveCheckKilledMemory" => "rl",
    "SlowProgress" => "sp",
    "NumericalFailure" => "er",
    "SkippedSolveCheck" => "sk",
    "SolveCheckCaughtError" => "other",
    )

residual_tol_satisfied(a, tol = 1e-5) = (all(isfinite, a) && maximum(a) < tol)
relative_tol_satisfied(a::T, b::T, tol::T = 1e-5) where {T <: Real} = (abs(a - b) / (1 + max(abs(a), abs(b))) < tol)

ex_wide_file(ex_name::String) = joinpath(output_folder, ex_name * "_wide.csv")

function make_wide_csv(ex_df, ex_name, ex_params)
    @info("making wide csv for $ex_name")
    inst_keys = ex_params[1]

    # add columns
    inst_ext_name(inst_set, extender) = (isempty(extender) ? inst_set : extender)
    transform!(ex_df, [:inst_set, :extender] => ((x, y) -> inst_ext_name.(x, y)) => :inst_ext)
    inst_solver_name(inst_ext, solver) = (inst_ext * "_" * solver)
    transform!(ex_df,
        [:inst_ext, :solver] => ((x, y) -> inst_solver_name.(x, y)) => :inst_solver,
        [:x_viol, :y_viol, :z_viol, :rel_obj_diff] => ByRow((res...) -> residual_tol_satisfied(coalesce.(res, NaN))) => :converged,
        [:status, :script_status] => ByRow((x, y) -> (y == "Success" ? status_map[x] : status_map[y])) => :status,
        )
    for (name, pos) in zip(inst_keys, ex_params[2])
        transform!(ex_df, :inst_data => ByRow(x -> eval(Meta.parse(x))[pos]) => name)
    end

    inst_solvers = unique(ex_df[:, :inst_solver])
    @info("instance solver combinations: $inst_solvers")

    # check objectives if solver claims optimality
    for group_df in groupby(ex_df, inst_keys)
        # check all pairs of verified converged results
        co_idxs = findall((group_df[:, :status] .== "co") .& group_df[:, :converged])
        if length(co_idxs) >= 2
            for i in eachindex(co_idxs)
                first_optval = group_df[co_idxs[i], :primal_obj]
                other_optvals = group_df[co_idxs[Not(i)], :primal_obj]
                if !all(relative_tol_satisfied.(other_optvals, first_optval))
                    println("objective values of: $(ex_name) $(group_df[!, :inst_data][1]) do not agree")
                end
            end
        end
    end

    unstacked_dims = [
        unstack(ex_df, inst_keys, :inst_ext, v, renamecols = x -> Symbol(v, :_, x), allowduplicates=true)
        for v in [:nu, :n, :p, :q]
        ]
    unstacked_res = [
        unstack(ex_df, inst_keys, :inst_solver, v, renamecols = x -> Symbol(v, :_, x), allowduplicates=true)
        for v in [:status, :converged, :iters, :solve_time]
        ]
    ex_df_wide = outerjoin(unstacked_dims..., unstacked_res..., on = inst_keys)
    CSV.write(ex_wide_file(ex_name), ex_df_wide)

    return (ex_df_wide, inst_solvers)
end

string_ast = "\$\\ast\$"
process_entry(::Missing) = string_ast
process_entry(::Missing, ::Missing) = "sk" # no data so instance was skipped
function process_entry(x::Float64)
    isnan(x) && return string_ast
    @assert x > 0
    if x < 10
        return @sprintf("%.1f", x)
    else
        return @sprintf("%.0f.", x)
    end
end
process_entry(st::String, converged::Bool) = (converged ? "\\underline{$(st)}" : st)
process_entry(x) = string(x)

function process_inst_solver(row, inst_solver)
    sep = " & "
    row_str = sep * process_entry(row[Symbol(:status_, inst_solver)], row[Symbol(:converged_, inst_solver)])
    row_str *= sep * process_entry(row[Symbol(:iters_, inst_solver)])
    row_str *= sep * process_entry(row[Symbol(:solve_time_, inst_solver)])
    return row_str
end

function make_table_tex(ex_name, ex_params, ex_df_wide, inst_solvers)
    @info("making table tex for $ex_name")
    inst_keys = ex_params[1]
    print_sizes = ex_params[4]

    sep = " & "
    ex_tex = open(joinpath(output_folder, ex_name * "_table.tex"), "w")
    header_str = prod(string(s) * sep for s in vcat(inst_keys, print_sizes))
    if print_table_solvers
        header_str *= prod(s * " & & & " for s in inst_solvers)
    end
    header_str = replace(header_str[1:(end - 2)], "_" => " ") * "\\\\"
    println(ex_tex, header_str)
    for row in eachrow(ex_df_wide)
        row_str = process_entry(row[1])
        for i in 2:length(inst_keys)
            row_str *= sep * process_entry(row[i])
        end
        for s in print_sizes
            row_str *= sep * process_entry(row[s])
        end
        if print_table_solvers
            for inst_solver in inst_solvers
                row_str *= process_inst_solver(row, inst_solver)
            end
        end
        row_str *= " \\\\"
        println(ex_tex, row_str)
    end
    close(ex_tex)

    return nothing
end

function transform_plot_cols(ex_df_wide, inst_solver)
    old_cols = Symbol.([:converged_, :solve_time_], inst_solver)
    transform!(ex_df_wide, old_cols => ByRow((x, y) -> ((!ismissing(x) && x) ? y : missing)) => inst_solver)
end

function make_plot_csv(ex_name, ex_params, ex_df_wide, inst_solvers)
    @info("making plot csv for $ex_name")
    inst_keys = ex_params[1]
    num_params = length(inst_keys)
    @assert 1 <= num_params <= 2 # handle case of more parameters if/when needed

    for inst_solver in inst_solvers
        transform_plot_cols(ex_df_wide, inst_solver)
    end

    plot_file_start = joinpath(output_folder, ex_name * "_plot")
    axis_name = last(inst_keys)
    if num_params == 1
        success_df = select(ex_df_wide, axis_name, inst_solvers...)
        CSV.write(plot_file_start * ".csv", success_df)
    else
        group_name = first(inst_keys)
        success_df = select(ex_df_wide, axis_name, group_name, inst_solvers...)
        for (group_id, group_df) in pairs(groupby(success_df, group_name))
            CSV.write(plot_file_start * "_$(group_id[1]).csv", select(group_df, Not(group_name)))
        end
    end

    return nothing
end

# run
post_process()
;
