using Printf
using CSV
using DataFrames

bench_file = joinpath(@__DIR__, "bench.csv")
output_folder = mkpath(joinpath(@__DIR__, "results"))

# uncomment examples to run
examples_params = Dict(
    "DensityEstJuMP"    => ([:m, :twod], [2, 3]), # TODO 2d?
    "ExpDesignJuMP"     => ([:logdetobj, :k], [5, 1]),
    "MatrixCompletionJuMP" => ([:d1, :d2], [1, 2]),
    "MatrixRegressionJuMP" => ([:m], [2]),
    "NearestPSDJuMP"    => ([:compl, :d], [2, 1]),
    "PolyMinJuMP"       => ([:m, :twod], [1, 2]),
    "PortfolioJuMP"     => ([:k], [1]),
    # "ShapeConRegrJuMP"  => ([:m, :twod], [1, 5]),
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
    # "KilledMemory" => "rl",
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
    for sub_df in groupby(ex_df, inst_keys)
        # check all pairs of converged results
        co_idxs = findall(sub_df[:status] .== "co")
        if length(co_idxs) >= 2
            for i in eachindex(co_idxs)
                first_optval = sub_df[co_idxs[i], :prim_obj]
                other_optvals = sub_df[co_idxs[Not(i)], :prim_obj]
                if !all(rel_tol_satisfied.(other_optvals, first_optval))
                    println("objective values of: $(ex_name) $(sub_df[:inst_data][1]) do not agree")
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

# TODO which make_plot_csv function?

function make_plot_csv(ex_name, ex_params)
    @info("making plot csv for $ex_name")
    ex_df_wide = CSV.read(ex_wide_file(ex_name))

    # TODO use examples_params dict instead of hard-coding conditions here
    if ex_name == "MatrixCompletionJuMP"
        transform!(ex_df_wide, [:d1, :d2] => ((x, y) -> round.(Int, y ./ x)) => :gr)
        x_var = :d1
    elseif ex_name in ["DensityEstJuMP", "PolyMinJuMP", "ShapeConRegrJuMP"]
        rename!(ex_df_wide, :m => :gr)
        x_var = :twod
    else
        ex_short = ex_name[1:(match(r"JuMP", ex_name).offset + 3)] # in case of suffix
        x_var = examples_params[ex_short][1][1]
        ex_df_wide.gr = fill("", nrow(ex_df_wide))
    end

    # TODO refac
    transform!(ex_df_wide, [:converged_nat_Hypatia, :solve_time_nat_Hypatia] => ByRow((x, y) -> (!ismissing(x) && x ? y : missing)) => :nat)
    transform!(ex_df_wide, [:converged_ext_Hypatia, :solve_time_ext_Hypatia] => ByRow((x, y) -> (!ismissing(x) && x ? y : missing)) => :ext)
    transform!(ex_df_wide, [:converged_ext_Mosek, :solve_time_ext_Mosek] => ByRow((x, y) -> (!ismissing(x) && x ? y : missing)) => :mosek)

    success_df = select(ex_df_wide, x_var, :gr, :nat, :ext, :mosek)
    for sub_df in groupby(success_df, :gr)
        plot_file = joinpath(output_folder, ex_name * "_plot_" * string(sub_df.gr[1]) * ".csv")
        CSV.write(plot_file, select(sub_df, Not(:gr)))
    end

    return nothing
end

# run
post_process()
;
