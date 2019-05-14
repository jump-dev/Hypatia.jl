#=
Copyright 2018, Chris Coey, Lea Kapelevich and contributors
=#

using DataFrames
import CSV
include(joinpath(@__DIR__(), "JuMP.jl"))

function scale_X!(X)
    X .-= 0.5 * (minimum(X, dims = 1) + maximum(X, dims = 1))
    X ./= (0.5 * (maximum(X, dims = 1) - minimum(X, dims = 1)))
    return nothing
end

# iris dataset
function iris_data()
    df = CSV.read(joinpath(@__DIR__, "data", "iris.csv"))
    # only use setosa species
    dfsub = df[df.species .== "setosa", [:sepal_length, :sepal_width, :petal_length, :petal_width]]
    X = convert(Matrix{Float64}, dfsub)
    scale_X!(X)
    n = 4
    return (X, n)
end

# lung cancer dataset from https://github.com/therneau/survival (cancer.rda)
# description at https://github.com/therneau/survival/blob/master/man/lung.Rd
function cancer_data()
    df = CSV.read(joinpath(@__DIR__, "data", "cancer.csv"), missingstring = "NA", copycols = true)
    dropmissing!(df, disallowmissing = true)
    # only use males with status 2
    dfsub = df[df.status .== 2, :]
    dfsub = dfsub[dfsub.sex .== 1, [:time, :age, :ph_ecog, :ph_karno, :pat_karno, :meal_cal, :wt_loss]]
    X = convert(Matrix{Float64}, dfsub)
    scale_X!(X)
    n = 7
    return (X, n)
end

function run_hard_densityest()
    degrees = 4:2:6

    datasets = [
        iris_data,
        cancer_data,
        ]

    for deg in degrees, s in datasets
        println()
        @show deg
        @show s
        println()

        (X, n) = s()
        dom = MU.Box(-ones(n), ones(n))
        (model,) = densityestJuMP(X, deg, use_monomials = true)

        (val, runtime, bytes, gctime, memallocs) = @timed JuMP.optimize!(model, JuMP.with_optimizer(HYP.Optimizer,
            use_dense = true,
            verbose = true,
            system_solver = SO.QRCholCombinedHSDSystemSolver,
            linear_model = MO.PreprocessedLinearModel,
            max_iters = 250,
            time_limit = 3.6e3,
            tol_rel_opt = 1e-5,
            tol_abs_opt = 1e-6,
            tol_feas = 1e-6,
            ))

        println()
        @show runtime
        @show bytes
        @show gctime
        @show memallocs
        println("\n\n")
    end
end

run_hard_densityest()
