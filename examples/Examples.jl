"""
Hypatia examples and script utilities.
"""
module Examples

using LinearAlgebra
# delete later, affects qr. see https://github.com/JuliaLang/julia/pull/40623
if VERSION < v"1.7.0-DEV.1188"
    const ColumnNorm = Val{true}
end

using Test
import Random
using Printf
import DataFrames
import CSV
import DataStructures: OrderedDict

import Hypatia
import Hypatia.PolyUtils
import Hypatia.Cones
import Hypatia.Models
import Hypatia.Solvers

include("common.jl")
include("common_native.jl")
include("common_JuMP.jl")
include("spectral_EFs_JuMP.jl")
include("setup.jl")

const model_types = [
    "native",
    "JuMP",
    ]

# list of names of native examples to run
const native_examples = [
    "densityest",
    "doptimaldesign",
    "linearopt",
    "matrixcompletion",
    "matrixregression",
    "maxvolume",
    "polyenvelope",
    "polymin",
    "portfolio",
    "sparsepca",
    ]

# list of names of JuMP examples to run
const JuMP_examples = [
    "CBLIB",
    "centralpolymat",
    "classicalquantum",
    "conditionnum",
    "contraction",
    "convexityparameter",
    "covarianceest",
    "densityest",
    "discretemaxlikelihood",
    "doptimaldesign",
    "entanglementassisted",
    "experimentdesign",
    "lotkavolterra",
    "lyapunovstability",
    "matrixcompletion",
    "matrixquadratic",
    "matrixregression",
    "maxvolume",
    "nearestcorrelation",
    "nearestpolymat",
    "nearestpsd",
    "nonparametricdistr",
    "normconepoly",
    "polyenvelope",
    "polymin",
    "polynorm",
    "portfolio",
    "regionofattr",
    "relentrentanglement",
    "robustgeomprog",
    "semidefinitepoly",
    "shapeconregr",
    "signomialmin",
    "sparselmi",
    "stabilitynumber",
    ]

load_example(mod::String, ex::String) =
    include(joinpath(@__DIR__, ex, mod * ".jl"))

get_test_instances(mod::String, ex::String) =
    include(joinpath(@__DIR__, ex, mod * "_test.jl"))

get_benchmark_instances(mod::String, ex::String) =
    include(joinpath(@__DIR__, ex, mod * "_benchmark.jl"))

model_type_examples(mod::String) = eval(Symbol(mod, "_examples"))

# load all examples
for mod in model_types, ex in model_type_examples(mod)
    load_example(mod, ex)
end

# build ordered dictionary of all test instances
function get_test_instances()
    test_insts = OrderedDict{String, OrderedDict}()
    for mod in model_types
        mod_insts = test_insts[mod] = OrderedDict()
        for ex in model_type_examples(mod)
            mod_insts[ex] = get_test_instances(mod, ex)
        end
    end
    return test_insts
end

end
