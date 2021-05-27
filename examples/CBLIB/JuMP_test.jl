
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

# other sets:

# cblib_power_small = [
#     "HMCR-n20-m400",
#     "HMCR-n100-m400",
#     "HMCR-n500-m400",
#     "HMCR-n20-m800",
#     "HMCR-n100-m800",
#     "HMCR-n500-m800",
#     ]
#
# cblib_exp_small = [
#     "bss1",
#     "demb782",
#     "bss2",
#     "demb781",
#     "synthes1",
#     "gptest",
#     "rijc781",
#     "synthes2",
#     "rijc784",
#     "rijc785",
#     "rijc786",
#     "syn05m",
#     "rijc782",
#     "batchdes",
#     "synthes3",
#     "rijc783",
#     "syn10m",
#     "syn05h",
#     "beck751",
#     "beck752",
#     "beck753",
#     "fiac81b",
#     "batch",
#     "syn05m02m",
#     "syn15m",
#     "fang88",
#     "syn10h",
#     "demb761",
#     "demb762",
#     "demb763",
#     "syn20m",
#     "syn05m02h",
#     "syn05m03m",
#     "fiac81a",
#     "syn30m",
#     "syn10m02m",
#     "rijc787",
#     "syn15h",
#     "ravem",
#     "syn05m04m",
#     "enpro56",
#     "syn05m03h",
#     "syn20h",
#     "syn40m",
#     "rsyn0805m",
#     "enpro48",
#     "syn10m02h",
#     "syn15m02m",
#     "syn10m03m",
#     "rsyn0810m",
#     "syn05m04h",
#     "rsyn0815m",
#     "syn30h",
#     "rsyn0820m",
#     "syn20m02m",
#     "rsyn0805h",
#     "syn10m04m",
#     "rsyn0830m",
#     "isil01",
#     "syn10m03h",
#     "syn15m02h",
#     "syn15m03m",
#     "rsyn0810h",
#     "syn40h",
#     "rsyn0840m",
#     "rsyn0815h",
#     "car",
#     "syn30m02m",
#     "syn20m02h",
#     "rsyn0820h",
#     "syn20m03m",
#     "syn10m04h",
#     "rsyn0805m02m",
#     "syn15m04m",
#     "rsyn0830h",
#     "rsyn0810m02m",
#     "syn15m03h",
#     "batchs101006m",
#     "syn40m02m",
#     "gp_dave_1",
#     "rsyn0815m02m",
#     "rsyn0840h",
#     "syn20m04m",
#     "syn30m02h",
#     "syn30m03m",
#     "rsyn0820m02m",
#     "syn20m03h",
#     "rsyn0805m03m",
#     "rsyn0805m02h",
#     "syn15m04h",
#     "jha88",
#     "batchs121208m",
#     "rsyn0830m02m",
#     "rsyn0810m02h",
#     "rsyn0810m03m",
#     "syn40m02h",
#     "syn40m03m",
#     "batchs151208m",
#     "syn30m04m",
#     "syn20m04h",
#     "varun",
#     "rsyn0840m02m",
#     "rsyn0815m02h",
#     ]
#
# cblib_exp_medium = [
#     "rsyn0815m03m",
#     "syn30m03h",
#     "rsyn0805m04m",
#     "gp_dave_2",
#     "rsyn0820m03m",
#     "rsyn0820m02h",
#     "rsyn0805m03h",
#     "batchs201210m",
#     "rsyn0810m04m",
#     "syn40m04m",
#     "rsyn0830m02h",
#     "rsyn0830m03m",
#     "rsyn0810m03h",
#     "rsyn0815m04m",
#     "syn40m03h",
#     "syn30m04h",
#     "LogExpCR-n20-m400",
#     "gp_dave_3",
#     "rsyn0820m04m",
#     "rsyn0815m03h",
#     "rsyn0840m02h",
#     "rsyn0840m03m",
#     "LogExpCR-n100-m400",
#     "rsyn0805m04h",
#     "rsyn0820m03h",
#     "LogExpCR-n500-m400",
#     "rsyn0810m04h",
#     "rsyn0830m04m",
#     "syn40m04h",
#     "rsyn0830m03h",
#     "rsyn0815m04h",
#     "rsyn0840m04m",
#     "rsyn0820m04h",
#     "rsyn0840m03h",
#     "mra01",
#     "rsyn0830m04h",
#     ]
