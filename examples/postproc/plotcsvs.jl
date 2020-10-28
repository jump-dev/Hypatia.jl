# TODO which make_plot_csv function?

function make_plot_csv(ex)
    @info("starting $ex")
    df = CSV.read(joinpath(@__DIR__, ex * "_wide.csv"))
    if ex == "MatrixCompletionJuMP"
        transform!(df, [:d1, :d2] => ((x, y) -> round.(Int, y ./ x)) => :gr)
        x_var = :d1
    elseif ex in ["DensityEstJuMP", "PolyMinJuMP", "ShapeConRegrJuMP"]
        rename!(df, :m => :gr)
        x_var = :twod
    else
        ex_short = ex[1:(match(r"JuMP", ex).offset + 3)] # in case of suffix
        x_var = params_map[ex_short][1][1]
        df.gr = fill("", nrow(df))
    end
    # TODO refac
    transform!(df, [:converged_nat_Hypatia, :solve_time_nat_Hypatia] => ByRow((x, y) -> (!ismissing(x) && x ? y : missing)) => :nat)
    transform!(df, [:converged_ext_Hypatia, :solve_time_ext_Hypatia] => ByRow((x, y) -> (!ismissing(x) && x ? y : missing)) => :ext)
    transform!(df, [:converged_ext_Mosek, :solve_time_ext_Mosek] => ByRow((x, y) -> (!ismissing(x) && x ? y : missing)) => :mosek)

    success_df = select(df, x_var, :gr, :nat, :ext, :mosek)
    for subdf in groupby(success_df, :gr)
        v = subdf.gr[1]
        CSV.write(joinpath(@__DIR__, ex * "$(v)_plot.csv"), select(subdf, Not(:gr)))
    end

    return success_df
end

# get_time(conv, time) = (coalesce(conv, false) ? time : missing)
#
# function make_plot_csv(ex)
#     @info("starting $ex")
#     df = CSV.read(joinpath(@__DIR__, ex * "_wide.csv"))
#     # only plot converged instances
#     # TODO remove missing here?
#     success_df = DataFrame(
#         x = Int[],
#         gr = Union{Int, String}[],
#         nat = Union{Missing, Float64}[],
#         ext = Union{Missing, Float64}[],
#         mosek = Union{Missing, Float64}[],
#         )
#
#     for r in eachrow(df)
#         nat_time = get_time(r.converged_nat_Hypatia, r.solve_time_nat_Hypatia)
#         ext_time = get_time(r.converged_ext_Hypatia, r.solve_time_ext_Hypatia)
#         mosek_time = get_time(r.converged_ext_Mosek, r.solve_time_ext_Mosek)
#
#         if ex == "MatrixCompletionJuMP"
#             param_cols = (r.d1, round(Int, r.d2 / r.d1))
#         elseif ex in ["DensityEstJuMP", "PolyMinJuMP", "ShapeConRegrJuMP"]
#             param_cols = (r.d, r.m)
#         else
#             ex_short = ex[1:(match(r"JuMP", ex).offset + 3)] # in case of suffix
#             x_var = params_map[ex_short][1][1]
#             param_cols = (r[x_var], "")
#         end
#
#         push!(success_df, (param_cols..., nat_time, ext_time, mosek_time))
#     end
#
#     for subdf in groupby(success_df, :gr)
#         v = subdf.gr[1]
#         CSV.write(joinpath(@__DIR__, ex * "$(v)_plot.csv"), select(subdf, Not(:gr)))
#     end
#
#     return success_df
# end

make_plot_csv.([
    "DensityEstJuMP",
    "ExpDesignJuMP",
    "MatrixCompletionJuMP",
    "MatrixRegressionJuMP",
    "NearestPSDJuMP",
    "PolyMinJuMP",
    "PortfolioJuMP",
    "ShapeConRegrJuMP",
    ])
;
