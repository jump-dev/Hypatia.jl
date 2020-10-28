using Printf
using CSV
using DataFrames

bench_file = joinpath(@__DIR__, "bench.csv")
output_folder = mkpath(joinpath(@__DIR__, "results"))

# uncomment examples to run
examples_params = Dict(
    "DensityEstJuMP"    => ([:m, :twod], [2, 3]), # TODO 2d?
    "ExpDesignJuMP"     => ([:logdetobj, :k], [5, 1]),
    "MatrixCompletionJuMP" => ([:k, :d], [1, 2]),
    "MatrixRegressionJuMP" => ([:m], [2]),
    "NearestPSDJuMP"    => ([:compl, :d], [2, 1]),
    "PolyMinJuMP"       => ([:m, :twod], [1, 2]),
    "PortfolioJuMP"     => ([:k], [1]),
    "ShapeConRegrJuMP"  => ([:m, :twod], [1, 5]),
    )
println("running examples:\n", keys(examples_params), "\n")

function post_process()
    @info("starting all")
    all_df = make_all_df()
    for (ex_name, ex_params) in examples_params
        @info("starting $ex_name with params: $ex_params")
        println()
        # uncomment functions to run for each example
        make_wide_csv(ex_name, ex_params, all_df)
        make_table_tex(ex_name, ex_params) # requires running make_wide_csv
        make_plot_csv(ex_name, ex_params) # requires running make_wide_csv
        @info("finished $ex_name")
    end
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

function make_wide_csv(ex_name, ex_params, all_df)
    @info("making wide csv for $ex_name")
    ex_df = all_df[all_df.example .== ex_name, :]

    # make columns out of tuple
    inst_keys = ex_params[1]
    for (name, pos) in zip(ex_params...)
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

function make_table_tex(ex_name, ex_params)
    @info("making table tex for $ex_name")
    ex_df_wide = CSV.read(ex_wide_file(ex_name))

    ex_tex = open(joinpath(output_folder, ex_name * "_table.tex"), "w")
    for r in eachrow(ex_df_wide)
        # TODO refac a bit
        # TODO print the size columns we want at start
        print(ex_tex,
            # nat_Hypatia
            process_entry(r.status_nat_Hypatia, r.converged_nat_Hypatia) * " & ",
            process_entry(r.iters_nat_Hypatia) * " & ",
            process_entry(r.solve_time_nat_Hypatia) * " & ",
            # ext_Hypatia
            process_entry(r.status_ext_Hypatia, r.converged_ext_Hypatia) * " & ",
            process_entry(r.iters_ext_Hypatia) * " & ",
            process_entry(r.solve_time_ext_Hypatia) * " & ",
            # ext_Mosek
            process_entry(r.status_ext_Mosek, r.converged_ext_Mosek) * " & ",
            process_entry(r.iters_ext_Mosek) * " & ",
            process_entry(r.solve_time_ext_Mosek) * " \\\\\n",
            )
    end
    close(ex_tex)

    return nothing
end

function transform_plot_cols(ex_df_wide, inst_solver::Symbol)
    old_cols = Symbol.([:converged_, :solve_time_], inst_solver)
    transform!(ex_df_wide, old_cols => ByRow((x, y) -> ((!ismissing(x) && x) ? y : missing)) => inst_solver)
end

inst_solvers = (:nat_Hypatia, :ext_Hypatia, :ext_Mosek)

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
