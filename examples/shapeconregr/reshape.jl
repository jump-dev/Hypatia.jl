using DataFrames
using CSV

for i in [1]
    longres = CSV.read("C:/Users/lkape/.julia/dev/Hypatia/examples/shapeconregr/shapeconregr_154402512.csv", comment="#")
    # longres = CSV.read(joinpath(@__DIR__(), "synthetic$(i).csv"), comment="#")
    longres = longres[[:fold, :signal_ratio, :deg, :use_wsos, :tm, :tr_rmse, :ts_rmse, :ignore_mono]]
    categorical!(longres, [:signal_ratio])

    longres_tm = longres[longres.ignore_mono .== true, :]
    longres_tm = longres_tm[[:fold, :signal_ratio, :deg, :use_wsos, :tm]]
    for subdf in groupby(longres, :use_wsos)
        wsos = subdf[:use_wsos][1]
        smry_res = aggregate(subdf, [:deg, :signal_ratio], [mean, minimum, maximum]) |> CSV.write("syn_$(i)_wsos_$(wsos).csv")
    end

    longres_er = longres[longres.use_wsos .== true, :]
    longres_er = longres_er[[:fold, :signal_ratio, :deg, :tr_rmse, :ts_rmse, :ignore_mono]]
    for subdf in groupby(longres_er, :ignore_mono)
        mono = subdf[:ignore_mono][1]
        smry_res = aggregate(subdf, [:deg, :signal_ratio], [mean, minimum, maximum]) |> CSV.write("syn_$(i)_mono_$(mono).csv")
    end
end
