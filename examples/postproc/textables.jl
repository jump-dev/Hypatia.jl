using Printf

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

function make_tex_table(ex)
    @info("starting $ex")
    df = CSV.read(joinpath(@__DIR__, ex * "_wide.csv"))
    io = open(joinpath(@__DIR__, ex * ".tex"), "w")
    for (i, r) in enumerate(eachrow(df))
        # TODO refac a bit
        # TODO print the size columns we want at start
        print(io,
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
    close(io)
    return df
end

make_tex_table.([
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
