
__precompile__()

module Alfonso
    import MathOptInterface
    const MOI = MathOptInterface
    using Printf
    using SparseArrays
    using LinearAlgebra
    
    include("optimizer.jl")
    # include("cones.jl")
    include("algorithm.jl")
end
