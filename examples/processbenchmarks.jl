using Printf
using CSV
using DataFrames

bench_file = joinpath(@__DIR__, "bench.csv")
output_folder = mkpath(joinpath(@__DIR__, "results"))

# uncomment examples to run
examples_params = Dict(
    "DensityEstJuMP" => (
        [:m, :deg], [2, 3],
        [:EP,], [:n_nat, :n_EP]
        # [:SEP,], [:n_nat, :n_SEP, :q_nat, :q_SEP]
        # [:EP, :SEP], [:n_nat, :n_EP, :n_SEP, :q_nat, :q_EP, :q_SEP]
        ),
    "ExpDesignJuMP" => (
        [:logdet, :k], [5, 1],
        # [:EP,], [:n_nat, :n_EP, :q_nat, :q_EP]
        [:SEP,], [:n_SEP, :q_nat, :q_SEP]
        # [:EP, :SEP], [:n_nat, :n_EP, :n_SEP, :q_nat, :q_EP, :q_SEP]
        ),
    "MatrixCompletionJuMP" => (
        [:k, :d], [1, 2],
        [:EP,], [:n_EP, :p_nat, :q_EP]
        # [:SEP,], [:n_nat, :n_SEP, :p_nat, :q_SEP]
        # [:EP, :SEP], [:n_nat, :n_EP, :n_SEP, :p_nat, :q_EP, :q_SEP]
        ),
    "MatrixRegressionJuMP" => (
        [:m], [2],
        [:SEP,], [:n_nat, :n_SEP, :p_SEP, :q_nat, :q_SEP]
        ),
    "NearestPSDJuMP" => (
        [:compl, :d], [2, 1],
        [:SEP,], [:n_nat, :q_SEP]
        ),
    "PolyMinJuMP" => (
        [:m, :halfdeg], [1, 2],
        [:SEP,], [:n_nat, :q_SEP]
        ),
    "PolyNormJuMP" => (
        [:L1, :n, :d, :m], [5, 1, 3, 4],
        [:SEP,], Symbol[]
        ),
    "PortfolioJuMP" => (
        [:k], [1],
        [:SEP,], Symbol[]
        ),
    "RandomPolyMatJuMP" => (
        [:n, :d, :m], [1, 2, 3],
        [:SEP,], Symbol[]
        ),
    "ShapeConRegrJuMP" => (
        [:m, :deg], [1, 5],
        [:SEP,], [:n_nat, :n_SEP, :q_nat]
        ),
    )

@info("running examples: $(keys(examples_params))")

function post_process()
    all_df = CSV.read(bench_file, DataFrame)
    replace!(all_df.extender, extender_map...)
    for (ex_name, ex_params) in examples_params
        println()
        ex_df = all_df[(all_df.example .== ex_name) .& in.(all_df.extender, Ref(vcat("", string.(ex_params[3])))), :]
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
    "SOCExpPSDOptimizer" => "SEP",
    "ExpPSDOptimizer" => "EP",
    )

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
        :status => ByRow(x -> status_map[x]) => :status,
        )
    for (name, pos) in zip(inst_keys, ex_params[2])
        transform!(ex_df, :inst_data => ByRow(x -> eval(Meta.parse(x))[pos]) => name)
    end

    inst_solvers = unique(ex_df[:inst_solver])
    @info("instance solver combinations: $inst_solvers")

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
        unstack(ex_df, inst_keys, :inst_ext, v, renamecols = x -> Symbol(v, :_, x))
        for v in [:n, :p, :q]
        ]
    unstacked_res = [
        unstack(ex_df, inst_keys, :inst_solver, v, renamecols = x -> Symbol(v, :_, x))
        for v in [:status, :converged, :iters, :solve_time]
        ]
    ex_df_wide = join(unstacked_dims..., unstacked_res..., on = inst_keys)
    CSV.write(ex_wide_file(ex_name), ex_df_wide)

    return (ex_df_wide, inst_solvers)
end

process_entry(::Missing) = "\$\\ast\$"
process_entry(::Missing, ::Missing) = "sk"
process_entry(x::Int) = (isnan(x) ? "\$\\ast\$" : string(x))
function process_entry(x::Float64)
    isnan(x) && return "\$\\ast\$"
    @assert x > 0
    # if x < 0.99
    #     str = @sprintf("%.2f", x)
    #     return str[2:end]
    # elseif x < 10
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
    header_str *= prod(s * " & & & " for s in inst_solvers)
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
        for inst_solver in inst_solvers
            row_str *= process_inst_solver(row, inst_solver)
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
