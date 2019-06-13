#=
Copyright 2019, Chris Coey, Lea Kapelevich and contributors

list of predefined datasets
=#

import DataFrames
import CSV

# iris dataset
function iris_data()
    df = CSV.read(joinpath(@__DIR__, "data", "iris.csv"))
    # only use setosa species
    dfsub = df[df.species .== "setosa", [:sepal_length, :sepal_width, :petal_length, :petal_width]] # n = 4
    X = convert(Matrix{Float64}, dfsub)
    return X
end

# lung cancer dataset from https://github.com/therneau/survival (cancer.rda)
# description at https://github.com/therneau/survival/blob/master/man/lung.Rd
function cancer_data()
    df = CSV.read(joinpath(@__DIR__, "data", "cancer.csv"), missingstring = "NA", copycols = true)
    DataFrames.dropmissing!(df, disallowmissing = true)
    # only use males with status 2
    dfsub = df[df.status .== 2, :]
    dfsub = dfsub[dfsub.sex .== 1, [:time, :age, :ph_ecog, :ph_karno, :pat_karno, :meal_cal, :wt_loss]] # n = 7
    X = convert(Matrix{Float64}, dfsub)
    return X
end
