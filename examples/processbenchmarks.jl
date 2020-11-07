using Printf
using CSV
using DataFrames

bench_file = joinpath(homedir(), "bench/bench_nocorr14.csv")
output_folder = mkpath(joinpath(@__DIR__, "results"))

# uncomment examples to run
all_dims = [:n_nat, :p_nat, :q_nat, :n_ext, :p_ext, :q_ext]
examples_params = Dict(
    # "DensityEstJuMP"    => ([:m, :deg], [2, 3], [:n_nat, :n_ext]),
    # "ExpDesignJuMP"     => ([:logdet, :k], [5, 1], [:n_nat, :q_nat, :q_ext]),
    # "MatrixCompletionJuMP" => ([:k, :d], [1, 2], [:n_nat, :p_nat, :q_ext]),
    # "MatrixRegressionJuMP" => ([:m], [2], all_dims),
    # "NearestPSDJuMP"    => ([:compl, :d], [2, 1], [:n_nat, :q_ext]),
    # "PolyMinJuMP"       => ([:m, :halfdeg], [1, 2], [:n_nat, :q_ext]),
    # "PolyNormJuMP"      => ([:L1, :n, :dr, :d, :m], [5, 1, 2, 3, 4], Symbol[]),
    # "PortfolioJuMP"     => ([:k], [1], Symbol[]),
    "RandomPolyMatJuMP" => ([:n, :d, :m], [1, 2, 3], Symbol[]),
    # "ShapeConRegrJuMP"  => ([:m, :deg], [1, 5], [:n_nat, :q_nat, :n_ext]),
    )

println("running examples:")
for k in keys(examples_params)
    println(k)
end

function post_process()
    all_df = make_all_df()
    inst_solvers = unique(all_df[:inst_solver])
    for (ex_name, ex_params) in examples_params
        println()
        @info("starting $ex_name with params: $ex_params")
        # uncomment functions to run for each example
        make_wide_csv(all_df, ex_name, ex_params)
        make_table_tex_polys(ex_name, ex_params, inst_solvers) # requires running make_wide_csv
        # make_plot_csv(ex_name, ex_params) # requires running make_wide_csv
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

residual_tol_satisfied(a, tol = 1e-6) = (all(isfinite, a) && maximum(a) < tol)
relative_tol_satisfied(a::T, b::T, tol::T = 1e-5) where {T <: Real} = (abs(a - b) / (1 + max(abs(a), abs(b))) < tol)

function make_all_df()
    all_df = DataFrame(CSV.File(bench_file))
    transform!(
        all_df,
        [:inst_set, :solver] => ((x, y) -> x .* "_" .* y) => :inst_solver,
        [:x_viol, :y_viol, :z_viol, :rel_obj_diff] => ByRow((res...) -> residual_tol_satisfied(res)) => :converged,
        :status => ByRow(x -> status_map[x]) => :status,
        )
    return all_df
end

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
                if !all(relative_tol_satisfied.(other_optvals, first_optval))
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
    if x < 0.99
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

function make_table_tex(ex_name, ex_params, inst_solvers)
    @info("making table tex for $ex_name")
    ex_df_wide = CSV.read(ex_wide_file(ex_name))
    inst_keys = ex_params[1]
    num_params = length(inst_keys)
    print_sizes = ex_params[3]

    sep = " & "
    ex_tex = open(joinpath(output_folder, ex_name * "_table.tex"), "w")
    for row in eachrow(ex_df_wide)
        row_str = process_entry(row[1])
        for i in 2:num_params
            row_str *= sep * process_entry(row[i])
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

function make_table_tex_polys(ex_name, ex_params, inst_solvers)
    @info("making table tex for $ex_name")
    ex_df_wide = CSV.read(ex_wide_file(ex_name))

    sep = " & "
    ex_tex = open(joinpath(output_folder, ex_name * "_table.tex"), "w")
    if ex_name == "PolyNormJuMP"
        df = filter!(:L1 => (x -> x), ex_df_wide)
    else
        df = ex_df_wide
    end
    for df1 in groupby(df, :n)
        print(ex_tex, "\\multirow{$(nrow(df1))}{*}{$(df1[1, :n])}\n")
        for df2 in groupby(df1, :d)
            print(ex_tex, sep, "\\multirow{$(nrow(df2))}{*}{$(df2[1, :d])}\n")
            add_sep = false
            for row in eachrow(df2)
                row_str = (add_sep ? sep : "")
                add_sep = true
                row_str *= sep * process_entry(row[:m])
                for inst_solver in inst_solvers
                    row_str *= process_inst_solver(row, inst_solver)
                end
                row_str *= " \\\\\n"
                print(ex_tex, row_str)
            end
            print(ex_tex, "\\cmidrule(lr){2-11}\n")
        end
        print(ex_tex, "\\cmidrule(lr){2-11}\n")
    end
    close(ex_tex)

    return nothing
end

function make_table_tex_polynorm_l2(ex_name, ex_params, inst_solvers)
    @info("making table tex for $ex_name")
    ex_df_wide = CSV.read(ex_wide_file(ex_name))

    sep = " & "
    ex_tex = open(joinpath(output_folder, ex_name * "_table.tex"), "w")
    for df1 in groupby(filter!(:L1 => (x -> !x), ex_df_wide), :n)
        print(ex_tex, "\\multirow{$(nrow(df1))}{*}{$(df1[1, :n])}\n")
        for df2 in groupby(df1, :dr)
            print(ex_tex, sep, "\\multirow{$(nrow(df2))}{*}{$(df2[1, :dr])}\n")
            add_sep = false
            for df3 in gropuby(df2, [:n, :dr, :m])
                row_str = (add_sep ? sep : "")
                add_sep = true
                for row in eachrow(df3)
                    row_str *= sep * process_entry(row[:m])
                    for inst_solver in inst_solvers
                        row_str *= sep * process_entry(row[Symbol(:status_, inst_solver)], row[Symbol(:converged_, inst_solver)])
                        row_str *= sep * process_entry(row[Symbol(:solve_time_, inst_solver)])
                    end
                    obj_diff = row[:prim_obj_Hypatia_ext] / row[:prim_obj_Hypatia_nat]
                    row_str *= sep * process_entry(obj_diff)
                end
                row_str *= " \\\\\n"
                print(ex_tex, row_str)
            end
            print(ex_tex, "\\cmidrule(lr){2-13}\n")
        end
        print(ex_tex, "\\cmidrule(lr){1-13}\n")
    end
    close(ex_tex)

    return nothing
end

function transform_plot_cols(ex_df_wide, inst_solver::Symbol)
    old_cols = Symbol.([:converged_, :solve_time_], inst_solver)
    transform!(ex_df_wide, old_cols => ByRow((x, y) -> ((!ismissing(x) && x) ? y : missing)) => inst_solver)
end

function make_plot_csv(ex_name, ex_params, inst_solvers)
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
