
# directory of CBLIB files
cblib_dir = joinpath(ENV["HOME"], "cblib/cblib.zib.de/download/all")
if !isdir(cblib_dir)
    @warn("CBLIB download folder not found")
    cblib_dir = joinpath(@__DIR__, "cblib_data")
    cblib_diverse = String[]
else
    cblib_diverse = [
        "expdesign_D_8_4", # psd, exp
        "port_12_9_3_a_1", # psd, soc, exp
        "tls4", # soc
        "ck_n25_m10_o1_1", # rsoc
        "rsyn0805h", # exp
        "2x3_3bars", # psd
        "HMCR-n20-m400", # power
        "classical_20_0", # soc, orthant
        "achtziger_stolpe06-6.1flowc", # rsoc, orthant
        "LogExpCR-n100-m400", # exp, orthant
    ]
end

relaxed_tols = (default_tol_relax = 1000,)
insts = OrderedDict()
insts["minimal"] = [(("expdesign_D_8_4",), nothing, relaxed_tols)]
insts["fast"] = [((inst,), nothing, relaxed_tols) for inst in cblib_diverse]
insts["various"] = insts["fast"]
return (CBLIBJuMP, insts)
