#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia
    using Printf
    using SparseArrays
    using LinearAlgebra
    using ForwardDiff
    using DiffResults

    include("interpolation.jl")
    include("cone.jl")
    for primcone in [
        "orthant",
        "dualsumofsquares",
        "secondorder",
        "exponential",
        "power",
        "rotatedsecondorder",
        "positivesemidefinite",
        "ellinfinity",
        ]
        include(joinpath(@__DIR__, "primitivecones", primcone * ".jl"))
    end
    include("linearsystem.jl")
    include("nativeinterface.jl")

    import MathOptInterface
    MOI = MathOptInterface
    include("mathoptinterface.jl")
end
