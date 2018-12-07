using DataFrames
using CSV

isdir("noise10") || mkdir("noise10")

for i in [1, 3, 5]
    # longres = CSV.read("C:/Users/lkape/.julia/dev/Hypatia/examples/shapeconregr/shapeconregr_154402512.csv", comment="#")
    longres = CSV.read(joinpath(@__DIR__(), "synthetic$(i).csv"), comment="#")
    longres = longres[[:fold, :signal_ratio, :deg, :use_wsos, :tm, :tr_rmse, :ts_rmse, :ignore_mono]]
    categorical!(longres, [:signal_ratio])

    longres_tm = longres[longres.ignore_mono .== false, :]
    longres_tm = longres_tm[[:fold, :signal_ratio, :deg, :use_wsos, :tm]]
    for subdf in groupby(longres_tm, :use_wsos)
        wsos = subdf[:use_wsos][1]
        smry_res = aggregate(subdf, [:deg, :signal_ratio], [median, minimum, maximum]) |> CSV.write("noise10/syn_$(i)_wsos_$(wsos).csv")
    end

    longres_er = longres[longres.use_wsos .== true, :]
    longres_er = longres_er[[:fold, :signal_ratio, :deg, :tr_rmse, :ts_rmse, :ignore_mono]]
    for subdf in groupby(longres_er, :ignore_mono)
        mono = subdf[:ignore_mono][1]
        smry_res = aggregate(subdf, [:deg, :signal_ratio], [median, minimum, maximum]) |> CSV.write("noise10/syn_$(i)_mono_$(mono).csv")
    end
end
