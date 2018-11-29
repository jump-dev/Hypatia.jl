#=
Copyright 2018, Chris Coey and contributors
=#

module Hypatia
    using Printf
    using LinearAlgebra
    using SparseArrays

    import FFTW
    import Combinatorics
    import GSL: sf_gamma_inc_Q
    import DynamicPolynomials
    import SemialgebraicSets
    # using Distributions
    include("interpolation.jl")

    import ForwardDiff
    import DiffResults
    # import PositiveFactorizations
    include("cone.jl")
    for primitivecone in [
        "orthant",
        "epinorminf",
        "epinormeucl",
        "epipersquare",
        "hypoperlog",
        "epiperpower",
        "hypogeomean",
        "epinormspectral",
        "semidefinite",
        "wsospolyinterp",
        "hypoperlogdet",
        "epipersumexp",
        ]
        include(joinpath(@__DIR__, "primitivecones", primitivecone * ".jl"))
    end

    # import IterativeSolvers
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
