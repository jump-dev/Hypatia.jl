#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia
    using Printf
    using LinearAlgebra
    using SparseArrays

    import FFTW
    import Combinatorics
    include("interpolation.jl")

    import ForwardDiff
    import DiffResults
    include("cone.jl")
    for primitivecone in [
        "orthant",
        "dualsumofsquares",
        "secondorder",
        "exponential",
        "power",
        "rotatedsecondorder",
        "positivesemidefinite",
        "ellinfinity",
        "spectralnorm",
        ]
        include(joinpath(@__DIR__, "primitivecones", primitivecone * ".jl"))
    end

    import IterativeSolvers
    include("linearsystem.jl")
    for linsyssolver in [
        "qrsymm",
        "naive",
        ]
        include(joinpath(@__DIR__, "linsyssolvers", linsyssolver * ".jl"))
    end

    include("nativeinterface.jl")

    import MathOptInterface
    const MOI = MathOptInterface
    include("mathoptinterface.jl")
end
