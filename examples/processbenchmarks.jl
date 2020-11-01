using Printf
using CSV
using DataFrames

bench_file = joinpath(@__DIR__, "bench.csv")
output_folder = mkpath(joinpath(@__DIR__, "results"))

# uncomment examples to run
all_dims = [:n_nat, :p_nat, :q_nat, :n_ext, :p_ext, :q_ext]
examples_params = Dict(
    "DensityEstJuMP"    => ([:m, :deg], [2, 3], [:n_nat, :n_ext]),
    "ExpDesignJuMP"     => ([:logdet, :k], [5, 1], [:n_nat, :q_nat, :q_ext]),
    "MatrixCompletionJuMP" => ([:k, :d], [1, 2], [:n_nat, :p_nat, :q_ext]),
    "MatrixRegressionJuMP" => ([:m], [2], all_dims),
    "NearestPSDJuMP"    => ([:compl, :d], [2, 1], [:n_nat, :q_ext]),
    "PolyMinJuMP"       => ([:m, :halfdeg], [1, 2], [:n_nat, :q_ext]),
    "PortfolioJuMP"     => ([:k], [1], Symbol[]),
    "ShapeConRegrJuMP"  => ([:m, :deg], [1, 5], [:n_nat, :q_nat, :n_ext]),
    )

inst_solvers = (:nat_Hypatia, :ext_Hypatia, :ext_Mosek) # TODO generate automatically for each example depending on data available

println("running examples:")
for k in keys(examples_params)
    println(k)
end

function post_process()
    all_df = make_all_df()
    for (ex_name, ex_params) in examples_params
        println()
        @info("starting $ex_name with params: $ex_params")
        # uncomment functions to run for each example
        make_wide_csv(all_df, ex_name, ex_params)
        make_table_tex(ex_name, ex_params) # requires running make_wide_csv
        make_plot_csv(ex_name, ex_params) # requires running make_wide_csv
        @info("finished $ex_name")
    end
    println()
    @info("finished all")
end

# TODO distinguish dying on model building vs solve/check
status_map = Dict(
    "Optimal" => "co",
    "KilledTime" => "tl",
    "TimeLimit" => "tl",
    "SlowProgress" => "sp",
    "NumericalFailure" => "er",
    "SkippedSolveCheck" => "sk",
    "SetupModelKilledTime" => "tl",
    "SolveCheckKilledTime" => "tl",
    "SetupModelKilledMemory" => "rl",
    "SolveCheckKilledMemory" => "rl",
    # TODO remove if not needed
    "SetupModelCaughtError" => "er",
    "SolveCheckCaughtError" => "er",
    )

function make_all_df()
    all_df = DataFrame(CSV.File(bench_file))
    transform!(
        all_df,
        [:inst_set, :solver] => ((x, y) -> x .* "_" .* y) => :inst_solver,
        [:x_viol, :y_viol, :z_viol, :rel_obj_diff] => ByRow((res...) -> (all(isfinite.(res)) && maximum(res) < 1e-6)) => :converged,
        :status => ByRow(x -> status_map[x]) => :status,
        )
    return all_df
end

rel_tol_satisfied(a, b) = (abs(a - b) / (1 + max(abs(a), abs(b))) < 1e-5)

ex_wide_file(ex_name::String) = joinpath(output_folder, ex_name * "_wide.csv")

function make_wide_csv(all_df, ex_name, ex_params)
    @info("making wide csv for $ex_name")
    ex_df = all_df[all_df.example .== ex_name, :]

    # make columns out of tuple
    inst_keys = ex_params[1]
    for (name, pos) in zip(inst_keys, ex_params[2])
        transform!(ex_df, :inst_data => ByRow(x -> eval(Meta.parse(x))[pos]) => name)
    end

    # check objectives if solver claims optimality
    for group_df in groupby(ex_df, inst_keys)
        # check all pairs of converged results
        co_idxs = findall(group_df[:status] .== "co")
        if length(co_idxs) >= 2
            for i in eachindex(co_idxs)
                first_optval = group_df[co_idxs[i], :prim_obj]
                other_optvals = group_df[co_idxs[Not(i)], :prim_obj]
                if !all(rel_tol_satisfied.(other_optvals, first_optval))
                    println("objective values of: $(ex_name) $(group_df[:inst_data][1]) do not agree")
                end
            end
        end
    end

    # TODO check that ext npq agrees for each formulation-instance
    unstacked_dims = [
        unstack(ex_df, inst_keys, :inst_set, v, renamecols = x -> Symbol(v, :_, x))
        for v in [:n, :p, :q]
        ]
    unstacked_res = [
        unstack(ex_df, inst_keys, :inst_solver, v, renamecols = x -> Symbol(v, :_, x))
        for v in [:status, :converged, :iters, :solve_time]
        ]
    ex_df_wide = join(unstacked_dims..., unstacked_res..., on = inst_keys)
    CSV.write(ex_wide_file(ex_name), ex_df_wide)

    return ex_df_wide
end

process_entry(::Missing) = "\$\\ast\$"
process_entry(::Missing, ::Missing) = "sk"
process_entry(x::Int) = (isnan(x) ? "\$\\ast\$" : string(x))
function process_entry(x::Float64)
    isnan(x) && return "\$\\ast\$"
    @assert x > 0
    if x < 1
        str = @sprintf("%.2f", x)
        return str[2:end]
    elseif x < 10
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

function make_table_tex(ex_name, ex_params)
    @info("making table tex for $ex_name")
    ex_df_wide = CSV.read(ex_wide_file(ex_name))
    inst_keys = ex_params[1]
    num_params = length(inst_keys)
    @assert 1 <= num_params <= 2 # handle case of more parameters if/when needed
    print_sizes = ex_params[3]

    sep = " & "
    ex_tex = open(joinpath(output_folder, ex_name * "_table.tex"), "w")
    for row in eachrow(ex_df_wide)
        row_str = process_entry(row[1])
        if num_params == 2
            row_str *= sep * process_entry(row[2])
        end
        for s in print_sizes
            row_str *= sep * process_entry(row[s])
        end
        for inst_solver in inst_solvers
            row_str *= process_inst_solver(row, inst_solver)
        end

        row_str *= " \\\\\n"
        print(ex_tex, row_str)
    end
    close(ex_tex)

    return nothing
end

function transform_plot_cols(ex_df_wide, inst_solver::Symbol)
    old_cols = Symbol.([:converged_, :solve_time_], inst_solver)
    transform!(ex_df_wide, old_cols => ByRow((x, y) -> ((!ismissing(x) && x) ? y : missing)) => inst_solver)
end

function make_plot_csv(ex_name, ex_params)
    @info("making plot csv for $ex_name")
    ex_df_wide = CSV.read(ex_wide_file(ex_name))
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
