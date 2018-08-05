
module Alfonso
    using Printf
    using SparseArrays
    using LinearAlgebra

    include("interpolation.jl")
    include("conedata.jl")
    include("nativeinterface.jl")
    include("algorithm.jl")
    # include("mathoptinterface.jl")
end
