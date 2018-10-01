
module Alfonso
    using Printf
    using SparseArrays
    using LinearAlgebra

    include("interpolation.jl")
    include("cone.jl")
    for primcone in [
        "orthant",
        "sumofsquares",
        "secondorder",
        "exponential",
        # "power",
        "rotatedsecondorder",
        "positivesemidefinite",
        "ellinfinity",
        ]
        include(joinpath(@__DIR__, "primitivecones", primcone * ".jl"))
    end
    include("nativeinterface.jl")

    import MathOptInterface
    MOI = MathOptInterface
    include("mathoptinterface.jl")
end
