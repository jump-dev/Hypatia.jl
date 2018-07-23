
__precompile__()

module Alfonso
    import MathOptInterface
    const MOI = MathOptInterface
    using Printf
    using SparseArrays
    using LinearAlgebra

    include("barriers.jl")
    include("optimizer.jl")
    include("algorithm.jl")
end
