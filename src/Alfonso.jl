
module Alfonso
    using Printf
    using SparseArrays
    using LinearAlgebra

    include("interpolation.jl")
    include("cone.jl")
    for primcone in ["nonnegative", "sumofsquares", "secondorder", "exponential", "power", "rotatedsecondorder", "positivesemidefinite"]
        include(joinpath(@__DIR__, "primitivecones", primcone * ".jl"))
    end
    include("nativeinterface.jl")
    # include("mathoptinterface.jl")
end
